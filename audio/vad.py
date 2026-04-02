"""
audio/vad.py

Silero VAD — Voice Activity Detection.
Tells us: "is this 30ms chunk speech or silence?"

Why Silero?
- Tiny PyTorch model (~1MB)
- Works on 16kHz mono audio
- Extremely fast — processes 30ms chunks in <1ms on CPU
- Much better than energy-based VAD (doesn't confuse noise with speech)

How it works:
- Feed it 30ms chunks one at a time
- It returns a probability (0-1) of speech presence
- We use a threshold (0.5) to decide speech vs silence
- We accumulate speech chunks into utterances
- When we see N consecutive silent chunks → utterance is complete
"""

from __future__ import annotations

import numpy as np
import torch
from config import (
    SAMPLE_RATE,
    VAD_THRESHOLD,
    SILENCE_CHUNKS,
)


class SileroVAD:
    def __init__(self):
        print("[VAD] Loading Silero VAD model...")
        # torch.hub downloads the model on first run (~1MB), cached after
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.model.eval()

        # Silero requires a hidden state that persists across chunks
        # We reset it at the start of each new utterance
        self._reset_state()
        print("[VAD] Silero VAD ready.")

    def _reset_state(self):
        """Reset hidden state — call this between utterances."""
        self.model.reset_states()

    def is_speech(self, chunk: np.ndarray) -> tuple[bool, float]:
        """
        Given a 30ms float32 chunk, returns (is_speech, confidence).

        chunk: np.ndarray of shape (CHUNK_SIZE,), dtype float32, range [-1, 1]
        """
        # Silero expects a 1D torch tensor
        tensor = torch.from_numpy(chunk).unsqueeze(0)  # shape: (1, CHUNK_SIZE)
        with torch.no_grad():
            prob = self.model(tensor, SAMPLE_RATE).item()
        return prob >= VAD_THRESHOLD, prob


class UtteranceDetector:
    """
    Wraps SileroVAD to accumulate chunks into complete utterances.

    State machine:
      WAITING  → no speech detected yet, discard chunks
      SPEAKING → speech detected, accumulate chunks
      SILENCE  → speech ended, count silent chunks
                 if silent long enough → emit utterance
                 if speech resumes → back to SPEAKING
    """

    def __init__(self):
        self.vad = SileroVAD()
        self._buffer = []          # accumulated audio chunks
        self._silence_count = 0    # consecutive silent chunks seen
        self._speaking = False     # are we currently in an utterance?

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        """
        Feed one 30ms chunk.
        Returns a complete utterance (np.ndarray) when one is detected, else None.
        """
        is_speech, prob = self.vad.is_speech(chunk)

        if is_speech:
            if not self._speaking:
                print(f"[VAD] Speech started (confidence: {prob:.2f})")
                self._speaking = True
                self._buffer = []
                self.vad._reset_state()

            self._buffer.append(chunk)
            self._silence_count = 0

        else:
            if self._speaking:
                self._buffer.append(chunk)   # include trailing silence in utterance
                self._silence_count += 1

                if self._silence_count >= SILENCE_CHUNKS:
                    # Silence long enough — utterance is complete
                    utterance = np.concatenate(self._buffer)
                    duration = len(utterance) / SAMPLE_RATE
                    print(f"[VAD] Utterance complete ({duration:.2f}s, {len(self._buffer)} chunks)")

                    self._speaking = False
                    self._buffer = []
                    self._silence_count = 0
                    self.vad._reset_state()

                    return utterance

        return None


if __name__ == "__main__":
    from audio.capture import AudioCapture

    print("Speak something — utterance will be detected and printed.")
    print("Ctrl+C to stop.\n")

    cap = AudioCapture()
    detector = UtteranceDetector()
    cap.start()

    try:
        while True:
            chunk = cap.read_chunk()
            utterance = detector.process_chunk(chunk)
            if utterance is not None:
                print(f"Got utterance: {len(utterance)} samples = {len(utterance)/SAMPLE_RATE:.2f}s")
                print("Waiting for next utterance...\n")
    except KeyboardInterrupt:
        cap.stop()
        print("Done.")
