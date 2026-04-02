"""
audio/capture.py

Handles microphone input using sounddevice.
Reads audio in small chunks that get fed into the VAD.

Why sounddevice?
- Simpler API than PyAudio
- Direct NumPy array output (no manual conversion needed)
- Works well on Linux, Mac, Windows
"""

import numpy as np
import sounddevice as sd
import queue
from config import SAMPLE_RATE, CHUNK_SIZE, CHANNELS


class AudioCapture:
    def __init__(self):
        self._queue = queue.Queue()
        self._stream = None
        self._running = False

    def _callback(self, indata, frames, time, status):  # noqa: ARG002
        """
        Called by sounddevice on every chunk of audio captured.
        indata shape: (CHUNK_SIZE, CHANNELS) — dtype float32
        We put it in a queue so the main thread can consume it.
        """
        if status:
            print(f"[AudioCapture] Warning: {status}")
        # indata is (frames, channels), take channel 0, copy to avoid buffer reuse
        self._queue.put(indata[:, 0].copy())

    def start(self):
        self._running = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SIZE,    # deliver exactly CHUNK_SIZE samples each time
            callback=self._callback,
        )
        self._stream.start()
        print(f"[AudioCapture] Listening at {SAMPLE_RATE}Hz, chunk={CHUNK_SIZE} samples")

    def read_chunk(self) -> np.ndarray:
        """
        Blocking call — returns the next 30ms chunk as float32 numpy array.
        Shape: (CHUNK_SIZE,)
        """
        return self._queue.get()

    def drain(self) -> None:
        """
        Discard all chunks currently queued.

        Why: while the agent is speaking via TTS, the mic is still capturing.
        Those chunks (containing the agent's own voice or ambient noise) would
        confuse the VAD if fed in after playback ends. Draining clears the
        backlog so the next listen cycle starts clean.
        """
        discarded = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                discarded += 1
            except Exception:
                break
        if discarded:
            print(f"[AudioCapture] Drained {discarded} stale chunks.")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[AudioCapture] Stopped.")


if __name__ == "__main__":
    # Quick test: capture 3 seconds of audio and print stats
    import time

    cap = AudioCapture()
    cap.start()

    chunks = []
    print("Recording for 3 seconds...")
    start = time.time()
    while time.time() - start < 3.0:
        chunk = cap.read_chunk()
        chunks.append(chunk)

    cap.stop()

    audio = np.concatenate(chunks)
    print(f"Captured {len(chunks)} chunks → {len(audio)} samples")
    print(f"Duration: {len(audio)/SAMPLE_RATE:.2f}s")
    print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    print("capture.py works correctly.")
