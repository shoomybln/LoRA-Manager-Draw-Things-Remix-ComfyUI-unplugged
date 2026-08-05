import sqlite3
import re
from safetensors import safe_open
from typing import Dict, List, Tuple
from .model_utils import determine_base_model
import os
import logging
import json

logger = logging.getLogger(__name__)

# Draw Things converts LoRA.safetensors to LoRA_f16.ckpt. Those .ckpt files are
# SQLite databases containing the quantized tensors. They do not embed the
# safetensors-style metadata header, so metadata is derived from the filename and
# the tensor keys.
_DRAW_THINGS_CKPT_SQLITE_HEADER = b"SQLite format 3\x00"


def _is_sqlite_ckpt(file_path: str) -> bool:
    """Return True if the file looks like a Draw Things SQLite-format .ckpt."""
    try:
        with open(file_path, "rb") as f:
            return f.read(16) == _DRAW_THINGS_CKPT_SQLITE_HEADER
    except OSError:
        return False


async def extract_lora_metadata(file_path: str) -> Dict:
    """Extract essential metadata from a LoRA file.

    Works for safetensors (embedded ``__metadata__``) and for Draw Things
    SQLite-format .ckpt files (metadata derived from tensor keys).
    """
    try:
        if file_path.lower().endswith(".safetensors"):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if metadata:
                    # Only extract base_model from ss_base_model_version
                    base_model = determine_base_model(metadata.get("ss_base_model_version"))
                    return {"base_model": base_model}
        elif file_path.lower().endswith(".ckpt") and _is_sqlite_ckpt(file_path):
            base_model = _detect_base_model_from_sqlite_ckpt(file_path)
            if base_model != "Unknown":
                return {"base_model": base_model}
    except Exception as e:
        logger.error(f"Error reading metadata from {file_path}: {str(e)}")
    return {"base_model": "Unknown"}


def _detect_base_model_from_sqlite_ckpt(file_path: str) -> str:
    """Detect the base model of a Draw Things SQLite .ckpt from its tensor keys.

    Draw Things quantized checkpoints expose the tensor names of the original
    model. Key prefixes commonly seen:
      - ``__te2__text_model__`` -> SDXL (has both CLIP encoders)
      - ``__text_model__`` -> SD1.5 / single-CLIP models
      - ``__dit__`` -> DiT-based models (Flux uses ``down_proj`` blocks,
        Qwen Image uses ``k``/``v``/``q`` attention blocks)
      - ``__unet__`` -> SD-family UNet-only LoRAs; SDXL has far more
        timestep blocks than SD1.5
    """
    try:
        conn = sqlite3.connect(f"file:{file_path}?mode=ro", uri=True)
        try:
            has_tensors = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tensors'"
            ).fetchone()
            if not has_tensors:
                return "Unknown"
            row = conn.execute(
                "SELECT name FROM tensors WHERE name LIKE '%te2__text_model__%' LIMIT 1"
            ).fetchone()
            if row:
                return "SDXL"
            row = conn.execute(
                "SELECT name FROM tensors WHERE name LIKE '%text_model__%' LIMIT 1"
            ).fetchone()
            if row:
                return "SD1.5"
            row = conn.execute(
                "SELECT name FROM tensors WHERE name LIKE '%dit%' OR name LIKE '%double_blocks%' LIMIT 1"
            ).fetchone()
            if row:
                return "Flux.1 D"
            row = conn.execute(
                "SELECT name FROM tensors WHERE name LIKE '%__unet__%' LIMIT 1"
            ).fetchone()
            if row:
                row = conn.execute(
                    "SELECT name FROM tensors WHERE name LIKE '%__unet__[t-%'"
                ).fetchall()
                max_t = 0
                for (name,) in row:
                    match = re.search(r"__unet__\[t-(\d+)", name)
                    if match:
                        max_t = max(max_t, int(match.group(1)))
                return "SDXL" if max_t > 600 else "SD1.5"
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error detecting base model from SQLite ckpt {file_path}: {e}")
    return "Unknown"

async def extract_checkpoint_metadata(file_path: str) -> dict:
    """Extract metadata from a checkpoint file to determine model type and base model"""
    try:
        # Analyze filename for clues about the model
        filename = os.path.basename(file_path).lower()
        
        model_info = {
            'base_model': 'Unknown',
            'model_type': 'checkpoint'
        }
        
        # Detect base model from filename
        if 'xl' in filename or 'sdxl' in filename:
            model_info['base_model'] = 'SDXL'
        elif 'sd3' in filename:
            model_info['base_model'] = 'SD3'  
        elif 'sd2' in filename or 'v2' in filename:
            model_info['base_model'] = 'SD2.x'
        elif 'sd1' in filename or 'v1' in filename:
            model_info['base_model'] = 'SD1.5'
        
        # Detect model type from filename
        if 'inpaint' in filename:
            model_info['model_type'] = 'inpainting'
        elif 'anime' in filename:
            model_info['model_type'] = 'anime'
        elif 'realistic' in filename:
            model_info['model_type'] = 'realistic'
        
        # Try to peek at the safetensors file structure if available
        if file_path.endswith('.safetensors'):
            import json
            import struct
            
            with open(file_path, 'rb') as f:
                header_size = struct.unpack('<Q', f.read(8))[0]
                header_json = f.read(header_size)
                header = json.loads(header_json)
                
                # Look for specific keys to identify model type
                metadata = header.get('__metadata__', {})
                if metadata:
                    # Try to determine if it's SDXL
                    if any(key.startswith('conditioner.embedders.1') for key in header):
                        model_info['base_model'] = 'SDXL'
                    
                    # Look for model type info
                    if metadata.get('modelspec.architecture') == 'SD-XL':
                        model_info['base_model'] = 'SDXL'
                    elif metadata.get('modelspec.architecture') == 'SD-3':
                        model_info['base_model'] = 'SD3'
                    
                    # Check for specific use case
                    if metadata.get('modelspec.purpose') == 'inpainting':
                        model_info['model_type'] = 'inpainting'
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error extracting checkpoint metadata for {file_path}: {e}")
        # Return default values
        return {'base_model': 'Unknown', 'model_type': 'checkpoint'}

async def extract_trained_words(file_path: str) -> Tuple[List[Tuple[str, int]], str]:
    """Extract trained words from a safetensors file and sort by frequency

    Args:
        file_path: Path to the safetensors file

    Returns:
        Tuple of:
        - List of (word, frequency) tuples sorted by frequency (highest first)
        - class_tokens value (or None if not found)
    """
    class_tokens = None

    if not file_path.lower().endswith(".safetensors"):
        return [], class_tokens

    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            
            # Extract class_tokens from ss_datasets if present
            if metadata and "ss_datasets" in metadata:
                try:
                    datasets_data = json.loads(metadata["ss_datasets"])
                    # Look for class_tokens in the first subset
                    if datasets_data and isinstance(datasets_data, list) and datasets_data[0].get("subsets"):
                        subsets = datasets_data[0].get("subsets", [])
                        if subsets and isinstance(subsets, list) and len(subsets) > 0:
                            class_tokens = subsets[0].get("class_tokens")
                except Exception as e:
                    logger.error(f"Error parsing ss_datasets for class_tokens: {str(e)}")
            
            # Extract tag frequency as before
            if metadata and "ss_tag_frequency" in metadata:
                # Parse the JSON string into a dictionary
                tag_data = json.loads(metadata["ss_tag_frequency"])
                
                # The structure may have an outer key (like "image_dir" or "img")
                # We need to get the inner dictionary with the actual word frequencies
                if tag_data:
                    # Get the first key (usually "image_dir" or "img")
                    first_key = list(tag_data.keys())[0]
                    words_dict = tag_data[first_key]
                    
                    # Sort words by frequency (highest first)
                    sorted_words = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)
                    return sorted_words, class_tokens
    except Exception as e:
        logger.error(f"Error extracting trained words from {file_path}: {str(e)}")
    
    return [], class_tokens