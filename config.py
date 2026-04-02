import os

# Audio
SAMPLE_RATE = 16000          # Hz — Whisper and Silero both expect 16kHz
CHUNK_DURATION_MS = 32       # ms per VAD chunk — Silero requires min 512 samples at 16kHz
CHUNK_SIZE = 512             # Silero's required window size for 16kHz audio
CHANNELS = 1                 # mono

# VAD
VAD_THRESHOLD = 0.5          # 0-1, higher = less sensitive
SILENCE_DURATION_MS = 800    # ms of silence before we consider utterance done
SILENCE_CHUNKS = int(SILENCE_DURATION_MS / CHUNK_DURATION_MS)  # = 26 chunks

# Whisper
WHISPER_MODEL = "small"      # tiny / base / small / medium
                             # small is the sweet spot: fast + accurate + multilingual

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "rag", "data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
