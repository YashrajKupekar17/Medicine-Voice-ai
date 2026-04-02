"""
test_layer2.py

Tests Layer 2 (TTS) in isolation — no mic needed.
Downloads voice models on first run.

Run: python test_layer2.py
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_synthesize_english():
    print("--- Test: English synthesis ---")
    from audio.tts import PiperTTS

    tts = PiperTTS()
    audio, sample_rate = tts.synthesize("Hello. The dose is 500 milligrams.", language="en")

    assert isinstance(audio, np.ndarray),      "Output must be numpy array"
    assert audio.dtype == np.float32,          "Output must be float32"
    assert audio.max() <= 1.0,                 "Audio must be in [-1, 1]"
    assert audio.min() >= -1.0,                "Audio must be in [-1, 1]"
    assert len(audio) > sample_rate * 0.5,     "Should be at least 0.5 seconds of audio"
    assert sample_rate in (16000, 22050),      "Sample rate should be 16kHz or 22kHz"

    duration = len(audio) / sample_rate
    print(f"  Output: {len(audio)} samples @ {sample_rate}Hz = {duration:.2f}s ✓")
    print(f"  Amplitude range: [{audio.min():.3f}, {audio.max():.3f}] ✓\n")


def test_language_fallback():
    print("--- Test: Unknown language falls back to English ---")
    from audio.tts import PiperTTS

    tts = PiperTTS()
    # 'fr' (French) is not in our registry — should fall back to English
    audio, sample_rate = tts.synthesize("Test fallback.", language="fr")
    assert len(audio) > 0, "Should still produce audio via fallback"
    print("  Unknown language 'fr' correctly fell back to English ✓\n")


def test_model_caching():
    print("--- Test: Voice model is cached after first load ---")
    from audio.tts import PiperTTS
    import time

    tts = PiperTTS()

    # First call — loads model
    t0 = time.time()
    tts.synthesize("First call.", language="en")
    first_call = time.time() - t0

    # Second call — uses cached model
    t0 = time.time()
    tts.synthesize("Second call.", language="en")
    second_call = time.time() - t0

    assert second_call < first_call, "Cached call should be faster than first load"
    print(f"  First call:  {first_call:.3f}s (includes model load)")
    print(f"  Second call: {second_call:.3f}s (cached) ✓\n")


def test_audio_loop():
    print("--- Test: Full audio loop — synthesize and play English ---")
    from audio.tts import PiperTTS

    tts = PiperTTS()
    print("  You should hear: 'Paracetamol dose is 500 milligrams every 6 hours'")
    tts.speak("Paracetamol dose is 500 milligrams every 6 hours.", language="en")
    print("  Playback complete ✓\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Layer 2 Tests — TTS")
    print("=" * 50 + "\n")

    test_synthesize_english()
    test_language_fallback()
    test_model_caching()
    test_audio_loop()

    print("=" * 50)
    print("All Layer 2 tests passed.")
    print("=" * 50)
    print()
    print("Next: run the full audio loop test:")
    print("  python audio/tts.py   ← hear English + Hindi synthesis")
