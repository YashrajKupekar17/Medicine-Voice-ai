"""
pipeline.py

The live voice pipeline — connects all 6 layers end to end.

Flow:
  AudioCapture   → raw 32ms chunks (Layer 1)
  UtteranceDetector → VAD-gated complete utterances (Layer 1)
  PharmacyKiosk.process_audio() →
      WhisperSTT     → transcript + language (Layer 1)
      OllamaLLM      → tool selection (Layer 4)
      DrugRetriever  → relevant drug chunks (Layer 3)
      OllamaLLM      → final answer (Layer 4)
  PharmacyKiosk.speak() →
      PiperTTS       → synthesized speech (Layer 2)
  AudioCapture.drain() → discard audio captured during TTS playback

States:
  LISTENING   → feeding chunks to VAD, waiting for speech
  PROCESSING  → utterance received, running STT + LLM
  SPEAKING    → TTS playing back the answer
"""

from __future__ import annotations

import time

from audio.capture import AudioCapture
from audio.vad import UtteranceDetector
from agent.kiosk import PharmacyKiosk


def run(text_only: bool = False) -> None:
    """
    text_only=False  → full voice pipeline (mic + speakers)
    text_only=True   → keyboard input, printed output (useful for debugging)
    """
    if text_only:
        _run_text_mode()
        return

    _run_voice_mode()


def _run_voice_mode() -> None:
    # ── Startup ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PharmacyKiosk — Voice Pipeline")
    print("=" * 55)
    print("Loading models (first run downloads weights)...\n")

    capture   = AudioCapture()
    detector  = UtteranceDetector()
    assistant = PharmacyKiosk(text_only=False)

    print("\n" + "=" * 55)
    print("  Ready. Speak your question.")
    print("  Ctrl+C to stop.")
    print("=" * 55 + "\n")

    capture.start()

    try:
        while True:
            # ── LISTENING ─────────────────────────────────────────────────
            chunk = capture.read_chunk()
            utterance = detector.process_chunk(chunk)

            if utterance is None:
                continue

            # ── PROCESSING ────────────────────────────────────────────────
            print("\n[Pipeline] Utterance detected — processing...")
            t_start = time.time()

            answer, language = assistant.process_audio(utterance)

            if not answer:
                print("[Pipeline] No answer generated. Listening again...\n")
                continue

            t_llm = time.time() - t_start
            print(f"[Pipeline] Answer ready in {t_llm:.1f}s [{language}]")
            print(f"[Pipeline] Answer: {answer}\n")

            # ── SPEAKING ──────────────────────────────────────────────────
            assistant.speak(answer, language)

            # Drain mic queue — discard audio captured while agent was speaking
            # (prevents the agent's own voice from re-triggering VAD)
            capture.drain()
            detector.vad._reset_state()

            print("[Pipeline] Listening...\n")

    except KeyboardInterrupt:
        print("\n[Pipeline] Stopping...")
    finally:
        capture.stop()
        print("[Pipeline] Goodbye.")


def _run_text_mode() -> None:
    """Keyboard-driven mode — same orchestrator, no audio hardware needed."""
    print("\n" + "=" * 55)
    print("  PharmacyKiosk — Text Mode")
    print("=" * 55)
    print("Type your question. 'reset' clears memory. 'quit' exits.\n")

    assistant = PharmacyKiosk(text_only=True)

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "reset":
            assistant.reset()
            print("Session cleared.\n")
            continue

        t0 = time.time()
        answer = assistant.process_text(query)
        elapsed = time.time() - t0

        print(f"Assistant ({elapsed:.1f}s): {answer}\n")
