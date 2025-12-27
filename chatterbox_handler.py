import os
import sys
import logging
import torch
import random
import numpy as np
import folder_paths
from huggingface_hub import hf_hub_download, snapshot_download

# Add src directory to path for local chatterbox import
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

CHATTERBOX_MODEL_SUBDIR = os.path.join("tts", "chatterbox")
CHATTERBOX_REPO_ID = "ResembleAI/chatterbox"

# Files for standard TTS model
CHATTERBOX_FILES_TO_DOWNLOAD = [
    "ve.safetensors", 
    "t3_cfg.safetensors", 
    "s3gen.safetensors", 
    "tokenizer.json", 
    "conds.pt"
]

# Files for multilingual TTS model
CHATTERBOX_MTL_FILES_TO_DOWNLOAD = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json"
]

DEFAULT_MODEL_PACK_NAME = "resembleai_default_voice"
DEFAULT_MTL_MODEL_PACK_NAME = "resembleai_multilingual"

def get_supported_languages():
    """Return dictionary of supported language codes and names."""
    return SUPPORTED_LANGUAGES.copy()

def get_chatterbox_model_pack_names():
    """Get list of available model packs."""
    chatterbox_models_base_path = os.path.join(folder_paths.models_dir, CHATTERBOX_MODEL_SUBDIR)
    if not os.path.isdir(chatterbox_models_base_path):
        os.makedirs(chatterbox_models_base_path, exist_ok=True)
        return [DEFAULT_MODEL_PACK_NAME, DEFAULT_MTL_MODEL_PACK_NAME]
    
    packs = [d for d in os.listdir(chatterbox_models_base_path) 
             if os.path.isdir(os.path.join(chatterbox_models_base_path, d))]
    
    # Ensure defaults are first if they exist
    if DEFAULT_MODEL_PACK_NAME in packs:
        packs.insert(0, packs.pop(packs.index(DEFAULT_MODEL_PACK_NAME)))
    if DEFAULT_MTL_MODEL_PACK_NAME in packs:
        packs.insert(1 if DEFAULT_MODEL_PACK_NAME in packs else 0, 
                     packs.pop(packs.index(DEFAULT_MTL_MODEL_PACK_NAME)))
    
    # Return defaults even if folders don't exist yet
    return packs if packs else [DEFAULT_MODEL_PACK_NAME, DEFAULT_MTL_MODEL_PACK_NAME]

def get_model_pack_path(model_pack_name):
    """Get full path for a model pack."""
    if not model_pack_name: 
        return None
    return os.path.join(folder_paths.models_dir, CHATTERBOX_MODEL_SUBDIR, model_pack_name)

def _download_file_from_hf(repo_id, filename, local_dir):
    """Download a single file from HuggingFace."""
    destination = os.path.join(local_dir, filename)
    if not os.path.exists(destination):
        logger.info(f"Downloading '{filename}' from '{repo_id}'...")
        try:
            hf_hub_download(
                repo_id=repo_id, 
                filename=filename, 
                local_dir=local_dir, 
                local_dir_use_symlinks=False, 
                resume_download=True
            )
            logger.info(f"Successfully downloaded '{filename}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to download '{filename}': {e}")
            if os.path.exists(destination + ".incomplete"): 
                os.remove(destination + ".incomplete")
            return False
    return True

def download_chatterbox_model_pack_if_missing(model_pack_name, is_multilingual=False):
    """Download model pack files if missing."""
    ckpt_dir = get_model_pack_path(model_pack_name)
    if not ckpt_dir:
        logger.warning(f"Invalid model pack name '{model_pack_name}', cannot download.")
        return False
    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    files_to_download = CHATTERBOX_MTL_FILES_TO_DOWNLOAD if is_multilingual else CHATTERBOX_FILES_TO_DOWNLOAD
    
    all_files_ok = all(
        _download_file_from_hf(CHATTERBOX_REPO_ID, f, ckpt_dir) 
        for f in files_to_download
    )
    
    if not all_files_ok:
        logger.error(f"Some files failed to download for model pack '{model_pack_name}'. Check logs.")
    return all_files_ok

def load_chatterbox_tts_model(model_pack_name, device):
    """Load TTS model for a given pack onto a specified device."""
    ckpt_dir = get_model_pack_path(model_pack_name)
    if not ckpt_dir:
        raise ValueError(f"Invalid model_pack_name: {model_pack_name}")
    
    if not download_chatterbox_model_pack_if_missing(model_pack_name, is_multilingual=False):
        logger.warning(f"Not all model files could be verified for '{model_pack_name}'. Loading may fail.")
        
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Model pack directory '{model_pack_name}' not found at '{ckpt_dir}'.")

    try:
        logger.info(f"Loading Chatterbox TTS model from {ckpt_dir} onto {device}")
        tts_model = ChatterboxTTS.from_local(ckpt_dir, device=device)
        return tts_model
    except Exception as e:
        logger.error(f"Error loading ChatterboxTTS from '{ckpt_dir}': {e}", exc_info=True)
        raise

def load_chatterbox_multilingual_tts_model(model_pack_name, device):
    """Load Multilingual TTS model for a given pack onto a specified device."""
    ckpt_dir = get_model_pack_path(model_pack_name)
    if not ckpt_dir:
        raise ValueError(f"Invalid model_pack_name: {model_pack_name}")
    
    if not download_chatterbox_model_pack_if_missing(model_pack_name, is_multilingual=True):
        logger.warning(f"Not all model files could be verified for '{model_pack_name}'. Loading may fail.")
        
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Model pack directory '{model_pack_name}' not found at '{ckpt_dir}'.")

    try:
        logger.info(f"Loading Chatterbox Multilingual TTS model from {ckpt_dir} onto {device}")
        mtl_tts_model = ChatterboxMultilingualTTS.from_local(ckpt_dir, device=device)
        return mtl_tts_model
    except Exception as e:
        logger.error(f"Error loading ChatterboxMultilingualTTS from '{ckpt_dir}': {e}", exc_info=True)
        raise

def load_chatterbox_vc_model(model_pack_name, device):
    """Load VC model for a given pack onto a specified device."""
    ckpt_dir = get_model_pack_path(model_pack_name)
    if not ckpt_dir:
        raise ValueError(f"Invalid model_pack_name: {model_pack_name}")
    
    # Try to download both standard and multilingual files
    # VC model files are the same
    if not download_chatterbox_model_pack_if_missing(model_pack_name, is_multilingual=False):
        download_chatterbox_model_pack_if_missing(model_pack_name, is_multilingual=True)
        
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Model pack directory '{model_pack_name}' not found at '{ckpt_dir}'.")

    try:
        logger.info(f"Loading Chatterbox VC model from {ckpt_dir} onto {device}")
        vc_model = ChatterboxVC.from_local(ckpt_dir, device=device)
        return vc_model
    except Exception as e:
        logger.error(f"Error loading ChatterboxVC from '{ckpt_dir}': {e}", exc_info=True)
        raise

def load_chatterbox_models(model_pack_name, device):
    """Load both TTS and VC models for a given pack onto a specified device."""
    tts_model = load_chatterbox_tts_model(model_pack_name, device)
    vc_model = load_chatterbox_vc_model(model_pack_name, device)
    return tts_model, vc_model

def set_chatterbox_seed(seed: int):
    """Set random seed for reproducible generation."""
    MAX_NUMPY_SEED = 2**32 - 1
    actual_seed_for_torch_random = random.randint(1, 0xffffffffffffffff) if seed == 0 else seed
    actual_seed_for_numpy = random.randint(1, MAX_NUMPY_SEED) if seed == 0 else (seed % MAX_NUMPY_SEED)
    torch.manual_seed(actual_seed_for_torch_random)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(actual_seed_for_torch_random)
    random.seed(actual_seed_for_torch_random)
    np.random.seed(actual_seed_for_numpy)
