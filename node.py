"""
ComfyUI nodes for Chatterbox Multilingual TTS.
Provides voice cloning and text-to-speech synthesis with 23 language support.
"""

import os
import random
import tempfile

import numpy as np
import torch
import torchaudio

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

# Global model cache
_MODEL_CACHE = {}


def get_device():
    """Get the appropriate device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_or_load_model(device=None):
    """Load and cache the Chatterbox model."""
    global _MODEL_CACHE

    if device is None:
        device = get_device()

    cache_key = f"chatterbox_mtl_{device}"

    if cache_key not in _MODEL_CACHE:
        from .chatterbox_handler import (
            load_chatterbox_multilingual_tts_model,
            DEFAULT_MTL_MODEL_PACK_NAME,
        )

        print(f"[Chatterbox] Loading multilingual model on {device}...")
        _MODEL_CACHE[cache_key] = load_chatterbox_multilingual_tts_model(
            DEFAULT_MTL_MODEL_PACK_NAME, device
        )
        print("[Chatterbox] Model loaded successfully.")

    return _MODEL_CACHE[cache_key]


def set_seed(seed: int, device: str):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class ChatterboxTTSNode:
    """
    Chatterbox Multilingual TTS Node.
    Generate speech from text with optional voice cloning from reference audio.
    Supports 23 languages including English, Korean, Japanese, Chinese, and more.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of languages for dropdown
        language_list = [
            f"{code} ({name})" for code, name in sorted(SUPPORTED_LANGUAGES.items())
        ]

        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello, this is a test of the Chatterbox text to speech system.",
                        "tooltip": "Text to synthesize into speech (max 300 characters)",
                    },
                ),
                "language": (
                    language_list,
                    {
                        "default": "en (English)",
                        "tooltip": "Language for text-to-speech synthesis",
                    },
                ),
                "exaggeration": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.25,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Speech expressiveness (0.5=neutral, higher=more expressive, extreme values may be unstable)",
                    },
                ),
                "cfg_weight": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "CFG/Pace weight. Set to 0 for language transfer to reduce accent from reference audio.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.05,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Randomness in generation (higher=more varied)",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "tooltip": "Random seed (0 for random generation)",
                    },
                ),
            },
            "optional": {
                "reference_audio": (
                    "AUDIO",
                    {
                        "tooltip": "Optional reference audio for voice cloning. If not provided, uses default voice.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/tts"
    DESCRIPTION = "Generate speech from text using Chatterbox Multilingual TTS. Supports 23 languages and optional voice cloning."

    def generate(
        self,
        text,
        language,
        exaggeration,
        cfg_weight,
        temperature,
        seed,
        reference_audio=None,
    ):
        # Extract language code from selection (e.g., "en (English)" -> "en")
        language_code = language.split(" ")[0]

        # Get device and model
        device = get_device()
        model = get_or_load_model(device)

        # Set seed for reproducibility
        if seed != 0:
            set_seed(seed, device)

        # Handle reference audio
        audio_prompt_path = None
        temp_file = None

        if reference_audio is not None:
            # ComfyUI AUDIO format: {"waveform": tensor, "sample_rate": int}
            waveform = reference_audio["waveform"]
            sample_rate = reference_audio["sample_rate"]

            # Save to temp file for the model
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_prompt_path = temp_file.name

            # Ensure waveform is in correct format [channels, samples]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)  # Remove batch dimension

            torchaudio.save(audio_prompt_path, waveform.cpu(), sample_rate)

        try:
            # Truncate text to max length
            text = text[:300]

            print(f"[Chatterbox] Generating audio for: '{text[:50]}...'")
            print(
                f"[Chatterbox] Language: {language_code}, Exaggeration: {exaggeration}, CFG: {cfg_weight}"
            )

            if audio_prompt_path:
                print("[Chatterbox] Using reference audio for voice cloning")
            else:
                print("[Chatterbox] Using default voice")

            # Generate audio
            wav = model.generate(
                text,
                language_id=language_code,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

            print("[Chatterbox] Audio generation complete.")

            # Convert to ComfyUI AUDIO format
            # wav shape: [1, samples], model.sr is sample rate
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # [samples] -> [1, samples]

            # Add batch dimension: [channels, samples] -> [batch, channels, samples]
            wav = wav.unsqueeze(0)

            return ({"waveform": wav, "sample_rate": model.sr},)

        finally:
            # Clean up temp file
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ChatterboxTTS": ChatterboxTTSNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterboxTTS": "Chatterbox TTS (Multilingual)",
}
