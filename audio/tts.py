"""
audio/tts.py

Text-to-Speech using Piper — fully local, edge-friendly.

Why Piper?
- Built specifically for edge/embedded (runs on Raspberry Pi 4)
- Uses ONNX Runtime for inference — no PyTorch needed, very fast
- ~2MB model for low quality, ~60MB for medium — tiny footprint
- Supports 30+ languages including Hindi, Tamil, Telugu

Why not XTTS?
- XTTS produces more natural speech and supports voice cloning
- But: 1.8GB model, needs GPU for real-time, overkill for edge
- Piper synthesizes 22kHz audio faster than real-time on CPU

Architecture:
  text + language
       ↓
  pick voice model for language (lazy load, cached after first use)
       ↓
  Piper ONNX inference → AudioChunk generator (streaming)
       ↓
  concatenate chunks → numpy float32 array
       ↓
  sounddevice playback

Voice models are downloaded from HuggingFace on first use (~2-60MB each).
Cached in models/tts/ — subsequent runs are instant.
"""

from __future__ import annotations

import urllib.request
import numpy as np
import sounddevice as sd
from pathlib import Path
from piper import PiperVoice

from config import MODELS_DIR

# ---------------------------------------------------------------------------
# Voice registry
# Maps language code → (model_name, huggingface_path)
#
# Quality tiers per model name: x_low / low / medium / high
#   x_low  →  ~2MB,   robot-like,  fastest    (good for RPi zero)
#   low    →  ~28MB,  acceptable,  fast        (good for RPi 4)
#   medium →  ~60MB,  natural,     real-time   (good for laptop demo)
#
# HuggingFace base: https://huggingface.co/rhasspy/piper-voices/resolve/main/
# ---------------------------------------------------------------------------
VOICE_REGISTRY: dict[str, tuple[str, str]] = {
    "en": (
        "en_US-lessac-medium",
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    ),
    "hi": (
        "hi_IN-pratham-medium",     # verified: exists on HuggingFace
        "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
    ),
    "ta": (
        "ta_IN-x_low",              # Tamil
        "ta/ta_IN/x_low/ta_IN-x_low.onnx",
    ),
}

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
TTS_MODELS_DIR = Path(MODELS_DIR) / "tts"
TTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LANGUAGE = "en"

# Whisper sometimes detects a closely related language instead of the one
# we have a voice for. Map those to the best available substitute.
# ur (Urdu) → hi: same Perso-Arabic script, mutually intelligible speech.
LANGUAGE_ALIASES: dict[str, str] = {
    "ur": "hi",   # Urdu → Hindi voice
    "mr": "hi",   # Marathi → Hindi voice (closest available)
    "ne": "hi",   # Nepali → Hindi voice (closest available)
}


class PiperTTS:
    def __init__(self):
        # Lazy-loaded voice cache: language_code → PiperVoice instance
        # We don't load all voices at startup — only load when first needed
        self._voices: dict[str, PiperVoice] = {}
        print("[TTS] PiperTTS initialized (voices load on first use)")

    def _download_model(self, model_name: str, hf_path: str) -> Path:
        """
        Download .onnx model + .json config from HuggingFace if not cached.
        Returns the local path to the .onnx file.

        Piper needs two files per voice:
          model.onnx       — the neural network weights
          model.onnx.json  — phoneme mappings, sample rate, speaker config
        """
        onnx_path = TTS_MODELS_DIR / f"{model_name}.onnx"
        json_path  = TTS_MODELS_DIR / f"{model_name}.onnx.json"

        if onnx_path.exists() and json_path.exists():
            return onnx_path

        print(f"[TTS] Downloading voice model '{model_name}'...")
        try:
            urllib.request.urlretrieve(f"{HF_BASE}/{hf_path}", onnx_path)
            urllib.request.urlretrieve(f"{HF_BASE}/{hf_path}.json", json_path)
            print(f"[TTS] Downloaded to {onnx_path}")
        except Exception as e:
            # Clean up partial downloads so next run retries cleanly
            onnx_path.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download '{model_name}': {e}") from e
        return onnx_path

    def _get_voice(self, language: str) -> PiperVoice:
        """
        Return a loaded PiperVoice for the given language code.
        Downloads model on first call, caches after that.
        Falls back to English if language not in registry.
        """
        if language in LANGUAGE_ALIASES:
            aliased = LANGUAGE_ALIASES[language]
            print(f"[TTS] '{language}' aliased to '{aliased}'")
            language = aliased

        if language not in VOICE_REGISTRY:
            print(f"[TTS] No voice for '{language}', falling back to English")
            language = DEFAULT_LANGUAGE

        if language not in self._voices:
            model_name, hf_path = VOICE_REGISTRY[language]
            try:
                onnx_path = self._download_model(model_name, hf_path)
                print(f"[TTS] Loading voice '{model_name}'...")
                self._voices[language] = PiperVoice.load(str(onnx_path))
                print(f"[TTS] Voice ready.")
            except Exception as e:
                print(f"[TTS] Could not load '{language}' voice: {e}")
                if language != DEFAULT_LANGUAGE:
                    print(f"[TTS] Falling back to English.")
                    return self._get_voice(DEFAULT_LANGUAGE)
                raise

        return self._voices[language]

    def synthesize(self, text: str, language: str = DEFAULT_LANGUAGE) -> tuple[np.ndarray, int]:
        """
        Convert text to audio.

        Returns:
            audio:       float32 numpy array, range [-1, 1]
            sample_rate: int — needed for correct playback speed
        """
        voice = self._get_voice(language)

        # synthesize() returns a generator of AudioChunk objects.
        # Each chunk contains audio_float_array (float32, [-1,1]) and sample_rate.
        # We concatenate all chunks into one array.
        #
        # Why a generator? For long text, Piper can yield chunks as it synthesizes
        # sentence by sentence — you could start playing before synthesis finishes.
        # For now we collect all chunks first (simpler), streaming is Layer 6.
        chunks = list(voice.synthesize(text))

        if not chunks:
            return np.zeros(1, dtype=np.float32), 22050

        audio = np.concatenate([c.audio_float_array for c in chunks])
        sample_rate = chunks[0].sample_rate   # typically 22050 Hz for Piper

        return audio, sample_rate

    def speak(self, text: str, language: str = DEFAULT_LANGUAGE) -> None:
        """
        Synthesize and play audio. Blocks until playback is complete.

        Why blocking?
        In the pipeline: agent speaks → user listens → user speaks → agent processes.
        These steps are sequential. Non-blocking playback would let the mic open
        while TTS is still playing, causing feedback/overlap.
        """
        print(f"[TTS] Speaking [{language}]: '{text}'")
        audio, sample_rate = self.synthesize(text, language)

        # sounddevice.play expects float32 in [-1, 1] — exactly what Piper gives us
        sd.play(audio, samplerate=sample_rate)
        sd.wait()   # block until playback finishes


if __name__ == "__main__":
    tts = PiperTTS()

    print("\n--- English ---")
    tts.speak("The dosage for paracetamol is 500 milligrams every 6 hours.", language="en")

    print("\n--- Hindi ---")
    tts.speak("पेरासिटामोल की खुराक हर 6 घंटे में 500 मिलीग्राम है।", language="hi")

    print("\nDone.")
