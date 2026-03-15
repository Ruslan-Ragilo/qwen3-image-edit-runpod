import os
import io
import json
import time
import uuid
import hashlib
import logging
import requests
import base64
import re
from typing import Tuple, Optional
from urllib.parse import urlparse
from datetime import timedelta
from PIL import Image
from PIL.Image import Image as PILImage
from pydantic import BaseModel, HttpUrl, validator
from minio import Minio
from minio.error import S3Error
import runpod
from dotenv import load_dotenv
import llama_cpp
from llama_cpp import Llama

# Load environment variables
load_dotenv()

# Configure logging with custom formatter for structured logging
class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        
        # Avoid adding multiple handlers if the logger already has handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            # Prevent propagation to root logger to avoid duplicate logs
            self.logger.propagate = False
    
    def _format_message(self, job_id: str, message: str, **kwargs) -> str:
        """Format log message with job ID and additional context"""
        base_msg = f"[Job: {job_id}] {message}"
        if kwargs:
            context = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
            return f"{base_msg} | {context}"
        return base_msg
    
    def info(self, job_id: str, message: str, **kwargs):
        self.logger.info(self._format_message(job_id, message, **kwargs))
        
    def warning(self, job_id: str, message: str, **kwargs):
        self.logger.warning(self._format_message(job_id, message, **kwargs))
        
    def error(self, job_id: str, message: str, **kwargs):
        self.logger.error(self._format_message(job_id, message, **kwargs))
        
    def debug(self, job_id: str, message: str, **kwargs):
        self.logger.debug(self._format_message(job_id, message, **kwargs))

logger = StructuredLogger(__name__)

# Environment variables
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_REGION = os.getenv("S3_REGION", "us-west-000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_SECURE = os.getenv("S3_SECURE", "true").lower() == "true"
S3_OBJECT_PREFIX = os.getenv("S3_OBJECT_PREFIX", "")
PRESIGN_EXPIRY = int(os.getenv("PRESIGN_EXPIRY", "86400"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "26214400"))  # 25 MB
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "120"))
HF_TOKEN = os.getenv("HF_TOKEN")

# GGUF model configuration
GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH", "/models/qwen-image-edit-q4_k_m.gguf")
GGUF_MODEL_URL = os.getenv("GGUF_MODEL_URL", "https://huggingface.co/Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF/resolve/main/Qwen-Rapid-AIO-NSFW-v18.1-Q4_K_M.gguf")

# Validate required environment variables
required_env_vars = [
    ("S3_ENDPOINT", S3_ENDPOINT),
    ("S3_ACCESS_KEY", S3_ACCESS_KEY),
    ("S3_SECRET_KEY", S3_SECRET_KEY),
    ("S3_BUCKET", S3_BUCKET),
]

for name, value in required_env_vars:
    if not value:
        raise ValueError(f"Environment variable {name} is required")

# Initialize MinIO client
minio_client = Minio(
    S3_ENDPOINT,
    access_key=S3_ACCESS_KEY,
    secret_key=S3_SECRET_KEY,
    region=S3_REGION,
    secure=S3_SECURE,
)

# Check if bucket exists
try:
    if not minio_client.bucket_exists(S3_BUCKET):
        raise ValueError(f"Bucket {S3_BUCKET} does not exist")
except S3Error as e:
    raise ValueError(f"Failed to access bucket {S3_BUCKET}: {e}")

# Input validation schema
class ImageEditInput(BaseModel):
    image_url: HttpUrl
    prompt: str
    negative_prompt: str = ""
    seed: Optional[int] = None
    num_inference_steps: int = 30
    guidance_scale: float = 1.5  # true_cfg_scale in GGUF models
    extra: dict = {}

    @validator("image_url")
    def validate_image_url(cls, v):
        parsed = urlparse(str(v))
        if parsed.scheme not in ["http", "https"]:
            raise ValueError("URL must be HTTP or HTTPS")
        return v

# Global model variable
model = None

def download_model(job_id: str):
    """Download GGUF model from Hugging Face if not exists"""
    if os.path.exists(GGUF_MODEL_PATH):
        logger.info(job_id, "Model already exists locally", path=GGUF_MODEL_PATH)
        return
    
    logger.info(job_id, "Model not found locally, downloading from Hugging Face", url=GGUF_MODEL_URL)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(GGUF_MODEL_PATH), exist_ok=True)
    
    try:
        # Download with progress
        response = requests.get(GGUF_MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        last_log_time = time.time()
        
        with open(GGUF_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every 5 seconds
                    current_time = time.time()
                    if total_size > 0 and current_time - last_log_time > 5:
                        percent = (downloaded / total_size) * 100
                        logger.info(job_id, f"Download progress: {percent:.1f}%")
                        last_log_time = current_time
        
        logger.info(job_id, "Model downloaded successfully", 
                   path=GGUF_MODEL_PATH,
                   size_bytes=downloaded)
        
    except Exception as e:
        logger.error(job_id, "Failed to download model", error=str(e))
        # Clean up partial download
        if os.path.exists(GGUF_MODEL_PATH):
            os.remove(GGUF_MODEL_PATH)
        raise

def load_model():
    """Load GGUF version of Qwen-Image-Edit using llama.cpp"""
    global model
    if model is None:
        job_id = "MODEL_INIT"
        logger.info(job_id, "Loading GGUF Qwen-Image-Edit model...")
        load_start = time.time()
        
        try:
            # Download model if needed
            download_model(job_id)
            
            # Check CUDA availability
            import torch
            if not torch.cuda.is_available():
                logger.error(job_id, "CUDA is not available. This application requires a GPU.")
                raise RuntimeError("CUDA is not available. This application requires a GPU.")
            
            logger.info(job_id, "CUDA is available", 
                       device_count=torch.cuda.device_count(),
                       device_name=torch.cuda.get_device_name(0))
            
            # Load model with llama.cpp
            logger.info(job_id, "Loading model with llama.cpp")
            
            # Determine optimal number of GPU layers (33 = all layers for 7B model)
            n_gpu_layers = 33  # Offload all layers to GPU
            
            model = Llama(
                model_path=GGUF_MODEL_PATH,
                n_ctx=4096,  # Context window
                n_gpu_layers=n_gpu_layers,
                n_threads=8,  # CPU threads for prompt processing
                verbose=False,
                use_mmap=True,
                use_mlock=False,  # Don't lock memory
                seed=42,  # Default seed for reproducibility
            )
            
            load_time = time.time() - load_start
            logger.info(job_id, "GGUF model loaded successfully", 
                       total_load_time=f"{load_time:.2f}s",
                       model_path=GGUF_MODEL_PATH,
                       n_gpu_layers=n_gpu_layers)
            
        except Exception as e:
            logger.error(job_id, "Failed to load GGUF model", error=str(e))
            raise
    return model

def sha256_hex(text: str) -> str:
    """Calculate SHA256 hash of a string and return hex representation"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def encode_image(job_id: str, image: PILImage, format: str, quality: int = 95) -> Tuple[bytes, str, str]:
    """Encode PIL image to bytes with specified format"""
    logger.debug(job_id, "Encoding image", format=format, quality=quality, mode=image.mode)
    encode_start = time.time()
    
    buffer = io.BytesIO()
    
    if format == "jpeg":
        # Convert RGBA to RGB if needed for JPEG
        if image.mode in ("RGBA", "LA", "P"):
            logger.debug(job_id, "Converting image mode for JPEG", from_mode=image.mode, to_mode="RGB")
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
            image = background
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        content_type = "image/jpeg"
        extension = "jpg"
    else:  # PNG
        image.save(buffer, format="PNG", optimize=True)
        content_type = "image/png"
        extension = "png"
    
    image_bytes = buffer.getvalue()
    buffer.close()
    
    encode_time = time.time() - encode_start
    logger.info(job_id, "Image encoded successfully", 
               bytes=len(image_bytes), 
               format=format,
               encode_time=f"{encode_time:.2f}s")
    return image_bytes, extension, content_type

def image_to_base64(image: PILImage) -> str:
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_image(base64_str: str) -> PILImage:
    """Convert base64 string to PIL image"""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))

def run_qwen_edit_gguf(job_id: str, model, image: PILImage, prompt: str, **kwargs) -> PILImage:
    """Run Qwen-Image-Edit GGUF model"""
    logger.info(job_id, "Running Qwen-Image-Edit GGUF", prompt=prompt)
    infer_start = time.time()
    
    try:
        # Extract parameters
        seed = kwargs.get("seed", None)
        cfg_scale = kwargs.get("guidance_scale", 1.5)
        num_inference_steps = kwargs.get("num_inference_steps", 30)
        
        # Convert input image to base64
        img_base64 = image_to_base64(image)
        logger.debug(job_id, "Image converted to base64", 
                    base64_length=len(img_base64))
        
        # Format prompt for Qwen-Image-Edit GGUF
        # The model expects a special format with image embedded
        # Format: <image>\n{prompt}\nEdit this image: <base64>
        formatted_prompt = f"<image>\n{prompt}\nEdit this image: {img_base64}"
        
        logger.debug(job_id, "Formatted prompt", 
                    prompt_length=len(formatted_prompt))
        
        # Generate with llama.cpp
        logger.info(job_id, "Starting GGUF inference", 
                   cfg_scale=cfg_scale,
                   steps=num_inference_steps,
                   seed=seed)
        
        # Set random seed if not provided
        if seed is None:
            import random
            seed = random.randint(1, 2**32 - 1)
            logger.debug(job_id, "Generated random seed", seed=seed)
        
        output = model.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            max_tokens=4096,  # Max output size (enough for base64 image)
            temperature=0.7,
            top_p=0.95,
            seed=seed,
        )
        
        logger.debug(job_id, "GGUF inference completed")
        
        # Extract response
        response_text = output['choices'][0]['message']['content']
        logger.debug(job_id, "Response received", 
                    response_length=len(response_text),
                    response_preview=response_text[:100] + "...")
        
        # Parse base64 from response
        # Try different patterns
        img_base64_result = None
        
        # Pattern 1: Markdown image format ![alt](base64)
        markdown_match = re.search(r'!\[.*?\]\((data:image/\w+;base64,([^)]+))', response_text)
        if markdown_match:
            img_base64_result = markdown_match.group(2)
            logger.debug(job_id, "Found base64 in markdown format")
        
        # Pattern 2: Direct base64 with data URI
        if not img_base64_result:
            data_uri_match = re.search(r'data:image/\w+;base64,([^"\')\s]+)', response_text)
            if data_uri_match:
                img_base64_result = data_uri_match.group(1)
                logger.debug(job_id, "Found base64 in data URI format")
        
        # Pattern 3: Plain base64 (assuming whole response is base64)
        if not img_base64_result:
            # Check if the whole response looks like base64
            if re.match(r'^[A-Za-z0-9+/=]+$', response_text.strip()):
                img_base64_result = response_text.strip()
                logger.debug(job_id, "Found plain base64 response")
        
        if not img_base64_result:
            raise ValueError(f"Could not extract base64 image from response: {response_text[:200]}")
        
        # Decode base64 to image
        result_image = base64_to_image(img_base64_result)
        
        infer_time = time.time() - infer_start
        logger.info(job_id, "GGUF inference completed successfully", 
                   inference_time=f"{infer_time:.2f}s",
                   image_mode=result_image.mode,
                   image_size=result_image.size)
        
        return result_image
        
    except Exception as e:
        logger.error(job_id, "Error during GGUF inference", error=str(e), exc_info=True)
        raise

def handler(event):
    """Main handler function for Runpod serverless"""
    start_time = time.time()
    job_id = event.get("id", str(uuid.uuid4()))
    
    logger.info(job_id, "Processing job")
    
    try:
        # Parse and validate input
        input_data = ImageEditInput(**event["input"])
        logger.info(job_id, "Input validated", 
                   prompt=input_data.prompt,
                   image_url=str(input_data.image_url),
                   num_inference_steps=input_data.num_inference_steps,
                   guidance_scale=input_data.guidance_scale)
        
        # Hash the image URL
        url_hash = sha256_hex(str(input_data.image_url))
        logger.debug(job_id, "URL hashed", url_hash=url_hash)
        
        # Timing variables
        download_time = 0
        infer_time = 0
        upload_time = 0
        
        # Download image
        logger.info(job_id, "Downloading image from URL")
        download_start = time.time()
        try:
            from diffusers.utils import load_image
            pil_image = load_image(str(input_data.image_url)).convert("RGB")
            width, height = pil_image.size
            download_time = time.time() - download_start
            logger.info(job_id, "Image downloaded successfully", 
                       width=width, 
                       height=height, 
                       mode=pil_image.mode,
                       download_time=f"{download_time:.2f}s")
            
        except Exception as download_error:
            logger.error(job_id, "Error downloading image", error=str(download_error))
            raise ValueError(f"Failed to download image: {str(download_error)}")
        
        # Run image editing with GGUF
        logger.info(job_id, "Starting image editing with GGUF")
        
        # Prepare parameters
        model_params = {
            "seed": input_data.seed,
            "num_inference_steps": input_data.num_inference_steps,
            "guidance_scale": input_data.guidance_scale,
        }
        
        # Add extra parameters if provided
        model_params.update(input_data.extra)
        
        # Generate random seed if not provided or invalid
        if model_params.get("seed") is None or model_params.get("seed") <= 0:
            import random
            model_params["seed"] = random.randint(1, 2**32 - 1)
            logger.info(job_id, "Generated random seed", seed=model_params["seed"])
        
        # Run inference
        infer_start = time.time()
        try:
            edited_image = run_qwen_edit_gguf(
                job_id,
                model,
                pil_image,
                input_data.prompt,
                **model_params
            )
            infer_time = time.time() - infer_start
            logger.info(job_id, "Image editing completed", 
                       inference_time=f"{infer_time:.2f}s")
            
        except Exception as e:
            logger.error(job_id, "Error during image editing", error=str(e))
            raise ValueError(f"Failed to edit image: {str(e)}")
        
        # Save result as PNG
        try:
            if edited_image.mode not in ("RGB", "L"):
                edited_image = edited_image.convert("RGB")
            
            result_filename = f"/tmp/{url_hash}_{job_id}.png"
            edited_image.save(result_filename)
            
            # Read the saved PNG file
            with open(result_filename, "rb") as f:
                result_bytes = f.read()
            
            result_ext = "png"
            result_content_type = "image/png"
            
        except Exception as save_error:
            logger.error(job_id, "Error saving result image", error=str(save_error))
            raise ValueError(f"Failed to save result image: {str(save_error)}")
        
        # Upload result to S3
        upload_start = time.time()
        try:
            result_key = f"{S3_OBJECT_PREFIX}results/{url_hash}/{job_id}.{result_ext}"
            minio_client.put_object(
                S3_BUCKET,
                result_key,
                io.BytesIO(result_bytes),
                len(result_bytes),
                content_type=result_content_type
            )
            
            # Remove the temporary file
            try:
                os.remove(result_filename)
                logger.debug(job_id, "Temporary file removed", filename=result_filename)
            except Exception as remove_error:
                logger.warning(job_id, "Failed to remove temporary file", error=str(remove_error))
            
            # Generate presigned URL
            presigned_url = minio_client.presigned_get_object(
                S3_BUCKET,
                result_key,
                expires=timedelta(seconds=PRESIGN_EXPIRY)
            )
            upload_time = time.time() - upload_start
            
        except Exception as upload_error:
            logger.error(job_id, "Error uploading result", error=str(upload_error))
            raise ValueError(f"Failed to upload result: {str(upload_error)}")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        logger.info(job_id, "Job completed successfully", 
                   total_time=f"{total_time:.2f}s")
        
        # Prepare response metadata
        meta = {
            "source_url": str(input_data.image_url),
            "url_sha256": url_hash,
            "model": "Qwen-Image-Edit-GGUF-Q4_K_M",
            "prompt": input_data.prompt,
            "seed": model_params["seed"],
            "num_inference_steps": input_data.num_inference_steps,
            "guidance_scale": input_data.guidance_scale,
            "runtime": {
                "latency_ms_total": int(total_time * 1000),
                "latency_ms_download": int(download_time * 1000),
                "latency_ms_infer": int(infer_time * 1000),
                "latency_ms_upload": int(upload_time * 1000),
            },
            "image": {
                "width": width,
                "height": height,
                "format": result_ext.upper()
            }
        }
        
        # Return success response
        return {
            "status": "success",
            "result": {
                "presigned_url": presigned_url,
                "bucket": S3_BUCKET,
                "object_key": result_key,
                "content_type": result_content_type,
                "expires_in": PRESIGN_EXPIRY,
                "meta": meta
            }
        }
        
    except Exception as e:
        logger.error(job_id, "Error processing job", error=str(e), exc_info=True)
        
        # Return error response
        error_type = "ModelError"
        if isinstance(e, ValueError):
            error_type = "BadInput" if "Invalid" in str(e) else "DownloadFailed"
        elif isinstance(e, S3Error):
            error_type = "StorageError"
            
        return {
            "status": "error",
            "error": {
                "type": error_type,
                "message": str(e),
                "details": {
                    "job_id": job_id
                }
            }
        }

# Load model at cold start
try:
    logger.info("MODEL_INIT", "Starting model loading at cold start")
    load_model()
    logger.info("MODEL_INIT", "Model loaded successfully at cold start")
except Exception as e:
    # Create a temporary logger instance for this error
    temp_logger = StructuredLogger(__name__)
    temp_logger.error("MODEL_INIT", "Failed to load model at startup", error=str(e))
    raise

# Start the Runpod serverless handler
if __name__ == "__main__":
    # Create a temporary logger instance for startup
    temp_logger = StructuredLogger(__name__)
    temp_logger.info("STARTUP", "Starting Runpod serverless handler for Qwen-Image-Edit GGUF")
    runpod.serverless.start({"handler": handler})
