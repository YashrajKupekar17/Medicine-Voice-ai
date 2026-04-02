"""
audio/stt.py

Speech-to-Text using faster-whisper.

Why faster-whisper over openai-whisper?
- Same accuracy, 4x faster inference
- Uses CTranslate2 under the hood (optimized C++ inference)
- Lower memory footprint
- Same model weights, compatible format

Models (downloaded automatically on first run):
  tiny   →  ~75MB  — fastest, less accurate
  base   → ~145MB  — good for English
  small  → ~465MB  — best tradeoff, good multilingual  ← we use this
  medium →  ~1.5GB — better accuracy, slower
  large  →  ~3GB   — best accuracy, needs GPU

Whisper auto-detects the language — we use this to route TTS later.
"""

import numpy as np
from faster_whisper import WhisperModel
from config import WHISPER_MODEL, MODELS_DIR


class WhisperSTT:
    def __init__(self):
        print(f"[STT] Loading faster-whisper '{WHISPER_MODEL}' model...")
        print("[STT] First run will download model weights. Subsequent runs use cache.")

        # device="cpu" for edge/offline — change to "cuda" if GPU available
        # compute_type="int8" — quantized, uses less RAM, nearly same accuracy
        self.model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            download_root=MODELS_DIR,
        )
        print("[STT] Whisper ready.\n")

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """
        Transcribe a complete utterance.

        audio: np.ndarray, float32, shape (N,), 16kHz mono, range [-1, 1]

        Returns:
            transcript: str — the transcribed text
            language:   str — detected language code ("en", "hi", "ta", etc.)
        """
        # initial_prompt biases beam search toward pharmacy/medicine vocabulary.
        # Whisper uses this as "previous context" — makes it more likely to
        # recognize brand names like Crocin, Dolo, Combiflam over homophones.
        PHARMACY_PROMPT = (
            "Pharmacy kiosk. Medicines: Paracetamol, Dolo 650, Crocin 500, "
            "Calpol Syrup, Ibuprofen, Brufen, Combiflam, Metformin, Glycomet, "
            "Omeprazole, Omez, Pan 40, Digene, Cetirizine, Cetzine, Alerid, "
            "ORS, Electral, Amoxicillin, Azithromycin, Zithromax, Azithral, "
            "Vitamin C, Limcee, Benadryl, Amlodipine, Amlovas."
        )

        # faster-whisper expects float32 numpy array at 16kHz
        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            language=None,        # None = auto-detect language
            condition_on_previous_text=False,  # better for short utterances
            vad_filter=False,     # we already did VAD ourselves
            initial_prompt=PHARMACY_PROMPT,
        )

        # segments is a generator — join all segment texts
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        language = info.language

        # Low-confidence language detection is unreliable for short utterances
        # (e.g. "Ya, one Crocin" gets flagged as Indonesian at 0.67).
        # Fall back to English — the most common language for this kiosk.
        if info.language_probability < 0.75 and language != "en":
            print(f"[STT] Low language confidence ({info.language_probability:.2f}), defaulting to 'en'")
            language = "en"

        print(f"[STT] Language: {language} (confidence: {info.language_probability:.2f})")
        print(f"[STT] Transcript: '{transcript}'")

        return transcript, language


if __name__ == "__main__":
    from audio.capture import AudioCapture
    from audio.vad import UtteranceDetector

    print("Speak something — it will be transcribed.")
    print("Ctrl+C to stop.\n")

    cap = AudioCapture()
    detector = UtteranceDetector()
    stt = WhisperSTT()

    cap.start()

    try:
        while True:
            chunk = cap.read_chunk()
            utterance = detector.process_chunk(chunk)

            if utterance is not None:
                transcript, language = stt.transcribe(utterance)
                print(f"\n>>> [{language.upper()}] {transcript}\n")
                print("Listening...\n")

    except KeyboardInterrupt:
        cap.stop()
        print("Done.")
