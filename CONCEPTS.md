# Medicine Assistant — Concepts & Component Guide

This document explains every technology used in this project conceptually —
what it is, how it works at a high level, and why it was chosen over alternatives.

---

## Table of Contents

1. [LangGraph — the Agent Framework](#1-langgraph--the-agent-framework)
2. [ReAct Pattern — how the agent thinks](#2-react-pattern--how-the-agent-thinks)
3. [Ollama + llama3.2:3b — the local LLM](#3-ollama--llama323b--the-local-llm)
4. [Tool Calling (bind_tools) — structured actions](#4-tool-calling-bind_tools--structured-actions)
5. [MemorySaver — conversation memory](#5-memorysaver--conversation-memory)
6. [SQLite — the inventory database](#6-sqlite--the-inventory-database)
7. [FAISS — semantic vector search](#7-faiss--semantic-vector-search)
8. [Sentence Transformers — text embeddings](#8-sentence-transformers--text-embeddings)
9. [Silero VAD — voice activity detection](#9-silero-vad--voice-activity-detection)
10. [Faster-Whisper — speech to text](#10-faster-whisper--speech-to-text)
11. [Piper TTS — text to speech](#11-piper-tts--text-to-speech)
12. [argostranslate — offline translation](#12-argostranslate--offline-translation)
13. [sounddevice — audio I/O](#13-sounddevice--audio-io)
14. [Why Everything Runs Locally](#14-why-everything-runs-locally)
15. [Architecture Decisions at a Glance](#15-architecture-decisions-at-a-glance)

---

## 1. LangGraph — the Agent Framework

### What is it?

LangGraph is a library for building stateful, multi-step AI agents as **directed graphs**.
Each node in the graph is a Python function. Edges define which node runs next based on
the current state. The graph keeps running until it reaches an end node.

Think of it like a flowchart where each box is a function and the arrows are conditions.

```
START → call_model → (has tool calls?) → tools → call_model → (done?) → END
```

### The State

The graph passes a single `state` dictionary between nodes. In this project:

```python
class PharmacyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`messages` is the full conversation history. Every node reads the current messages
and appends new ones. The `add_messages` reducer ensures new messages are appended,
not overwritten.

### Why LangGraph over plain LangChain?

Plain LangChain chains are linear (A → B → C). LangGraph adds loops and conditionals.
An AI agent needs to loop — it calls a tool, gets the result, decides if it needs
another tool, and so on. LangGraph makes this loop explicit and inspectable.

**LangGraph also gives you:**
- Persistent state between steps (the message list stays intact across the loop)
- Checkpointing (save and resume conversations via MemorySaver)
- LangGraph Studio support (visual debugger)

---

## 2. ReAct Pattern — how the agent thinks

### What is ReAct?

ReAct stands for **Reasoning + Acting**. It is a prompting pattern where the LLM
alternates between two things:

1. **Reasoning** — "The customer said they have a headache. I should look up medicines for headaches."
2. **Acting** — Call the `drug_lookup` tool with the query "headache".

The result comes back, the LLM reasons about it again, and either calls another tool
or gives the final answer. This loop continues until the answer is complete.

```
Human: "I have a headache"
   ↓
Thought: customer has a symptom → use drug_lookup
   ↓
Action: drug_lookup("headache")
   ↓
Observation: "Paracetamol — available brands: Dolo 650, Crocin 500"
   ↓
Answer: "For headache I recommend Paracetamol. We have Dolo 650 and Crocin 500."
```

### Why this project skips LLM Round 2

In a classic ReAct loop, the LLM sees the tool result and generates the final answer.
This project **formats the answer directly in Python** after the tool runs (Round 2),
without asking the LLM again.

Why? Because `llama3.2:3b` is a small model (3 billion parameters). When given tool
results, it often hallucinates extra details ("we also have Aspirin at ₹20...") or
rephrases things incorrectly. The tool output already contains everything the customer
needs — passing it through the LLM again only adds risk.

The tradeoff: answers are less conversational but 100% accurate to the database.

---

## 3. Ollama + llama3.2:3b — the local LLM

### What is Ollama?

Ollama is a tool that lets you run large language models locally on your laptop.
It downloads and manages model files, and exposes an HTTP API (port 11434) that is
fully compatible with the OpenAI API format.

This means you can use the same `openai` Python client or LangChain's `ChatOllama`
connector and it just works — no API keys, no internet required at inference time.

### What is llama3.2:3b?

Llama 3.2 is Meta's open-source language model family. The `3b` variant has
3 billion parameters. It is quantized (compressed) when run via Ollama to use
approximately 2GB of RAM.

Key properties relevant to this project:
- **Supports native tool calling** — can output structured JSON tool calls,
  not just free text. This is the main reason it was chosen over phi3:mini.
- **Fast on CPU** — 3B parameters is small enough to run without a GPU.
- **Multilingual** — understands Hindi queries reasonably well even when responding in English.

### Why not GPT-4 or Claude?

This kiosk is designed for pharmacies that may not have reliable internet.
Cloud APIs introduce latency (300-1000ms per request), cost per query, and a
hard dependency on external services. A local LLM runs offline, sub-200ms
per token on modern hardware, and costs nothing per query.

### Why not phi3:mini?

`phi3:mini` was the first model tried. It does not support structured tool calling —
it outputs tool invocations as free text in formats like:
```
ACTION: add_to_cart(Dolo 650, 1)
```
This required fragile regex parsing with many edge cases. `llama3.2:3b` outputs
proper JSON tool calls natively, which LangChain can parse reliably.

---

## 4. Tool Calling (bind_tools) — structured actions

### What is tool calling?

Tool calling is a feature where the LLM, instead of writing a prose answer, outputs
a structured request to call a specific function with specific arguments:

```json
{
  "name": "add_to_cart",
  "args": { "brand_name": "Dolo 650", "quantity": 2 }
}
```

The LLM has been fine-tuned to output this JSON when it recognizes that the user's
request requires calling a tool. The application reads this JSON, executes the real
function, and feeds the result back to the LLM.

### How bind_tools works

`llm.bind_tools(TOOLS)` sends the tool schemas (names, parameter types, descriptions)
to the LLM as part of every request. The LLM uses this schema to know what tools exist
and what arguments they expect.

```python
_llm_with_tools = ChatOllama(model="llama3.2:3b").bind_tools(TOOLS)
response = _llm_with_tools.invoke(messages)

if response.tool_calls:
    # LLM wants to call a tool
    # response.tool_calls = [{"name": "...", "args": {...}}]
```

### Why keyword overrides exist

Even with native tool calling, `llama3.2:3b` occasionally skips tool calls and
responds with plain text ("Sorry, I don't have that information"). This happens for:
- Ambiguous purchase intent ("get me that one")
- Hindi/mixed language queries
- Very short queries ("bill please")

The keyword override system is a safety net: if the LLM returns no tool call but
the message contains "bill", "receipt", "checkout", it forces `generate_receipt`.
If it contains "headache", "fever", "cough", it forces `drug_lookup`. This ensures
critical actions always fire.

### Arg normalization

`llama3.2:3b` sometimes returns malformed args:

| What LLM sent | What it meant | Fix applied |
|---|---|---|
| `{"Dolo 650": 1}` | brand_name="Dolo 650" | positional fallback |
| `{"brand_name": {"Dolo": "string"}}` | brand_name="Dolo" | extract first dict key |
| `{"brand_name": ["Dolo 650", "Crocin"]}` | two items | expand into two calls |
| `{"quantity": "1"}` | quantity=1 | coerce string to int |

`_normalize_args()` and `_expand_list_args()` handle all of these before the
tool is executed.

---

## 5. MemorySaver — conversation memory

### The problem

An AI agent has no memory by default. Every message you send is treated as if
it is the first message. This makes it useless for multi-turn conversations —
the agent forgets what the customer just said.

### What MemorySaver does

`MemorySaver` is LangGraph's in-memory checkpointer. Every time the graph runs,
it saves the full message history under a `thread_id`. On the next invocation
with the same `thread_id`, the saved history is loaded and prepended to the new
message.

```python
# Same thread_id → same conversation history
graph.invoke(
    {"messages": [HumanMessage("I have a headache")]},
    config={"configurable": {"thread_id": "session-abc123"}}
)

# Later in the same session:
graph.invoke(
    {"messages": [HumanMessage("Add it to my cart")]},
    config={"configurable": {"thread_id": "session-abc123"}}
)
# The graph sees both messages — it knows "it" = Paracetamol
```

### Session management

Each customer gets a new `thread_id` (random UUID). When `reset()` is called:
- A new `thread_id` is created → fresh conversation history
- `clear_cart()` is called → empty cart

This means the cart and conversation history always stay in sync per session.

### Why not a database-backed checkpointer?

`MemorySaver` stores history in RAM. On process restart, all history is lost.
For a kiosk that handles one customer at a time and resets between customers,
this is intentional — there is no requirement to persist past sessions.
A `SqliteSaver` or `PostgresSaver` would be needed for multi-device or
persistent history scenarios.

---

## 6. SQLite — the inventory database

### What is SQLite?

SQLite is an embedded relational database. Unlike MySQL or PostgreSQL, it has
no server process — the entire database is a single `.db` file on disk.
Python includes it in the standard library (`import sqlite3`).

### Why SQL for inventory (not a vector DB)?

Inventory queries are **exact and transactional**:
- "Do you have Dolo 650?" → needs exact stock count, not a similarity score
- "Add 2 strips of Crocin 500" → needs atomic read-write (check stock → update cart)
- "Generate receipt" → needs to sum prices exactly

A vector database (like FAISS) finds things that are *similar* to a query.
For inventory, "similar" is not useful — you need exactly "Dolo 650", not
something that sounds like it.

SQL handles this perfectly with a simple `WHERE brand_name LIKE ?` query.

### The two tables

**`inventory`** — one row per brand:
```
Dolo 650 | Paracetamol | 650mg | tablet | 80 strips | ₹32 | OTC | Best Seller
Crocin 500 | Paracetamol | 500mg | tablet | 150 strips | ₹25 | OTC | —
```

**`drug_info`** — one row per generic medicine:
```
Paracetamol | Analgesic | Uses: headache, fever, pain | Dosage: 500-1000mg every 6h
```

The two tables are joined by `medicine_name = generic_name`. This separation
means drug medical information (which rarely changes) is stored separately
from stock and pricing (which changes frequently).

### Schedule H warning

Some medicines (`schedule = 'H'`) require a prescription in India.
The receipt generator detects these and prints a warning:
```
⚠ Prescription required for: Glycomet 500, Amoxil 500
  Please show prescription at counter.
```

---

## 7. FAISS — semantic vector search

### What is FAISS?

FAISS (Facebook AI Similarity Search) is a library for finding vectors that
are close to a query vector in high-dimensional space. Given a query like
"medicine for stomach pain", FAISS finds the stored text chunks most similar
to that query — even if they don't share exact words.

### How it works

1. **Ingestion (one-time):** Each drug description is converted to a 384-dimensional
   vector (a list of 384 floating point numbers) using a text embedding model.
   These vectors are stored in a FAISS index file on disk.

2. **At query time:** The user's query is also converted to a 384-dim vector.
   FAISS calculates the cosine similarity between the query vector and every stored
   vector and returns the closest matches.

### Why it exists alongside SQLite

SQLite's `LIKE "%fever%"` only works if the word "fever" appears literally in the
database. It cannot understand that "pyrexia" is another word for fever, or that
"high temperature" is related to fever.

FAISS understands semantic meaning — "elevated body temperature" will still match
"fever" because the embedding model maps related concepts to nearby vectors.

FAISS is used as a **fallback** when the SQL keyword search finds nothing.

### IndexFlatIP

This project uses `faiss.IndexFlatIP` — Flat (brute-force) index with Inner Product
(dot product = cosine similarity for normalized vectors). Brute-force is fine here
because the index has only ~23 drug entries. FAISS's approximate methods (IVF, HNSW)
give speed gains for millions of vectors — not needed at this scale.

---

## 8. Sentence Transformers — text embeddings

### What are embeddings?

An embedding model converts text into a dense vector (array of numbers) that
captures the semantic meaning of the text. Two sentences with similar meaning
will produce vectors that are close to each other in the vector space.

```
"I have a fever"          → [0.12, -0.34, 0.87, ...]  (384 numbers)
"high body temperature"   → [0.11, -0.31, 0.89, ...]  (very similar)
"buy a car"               → [-0.78, 0.45, -0.23, ...]  (very different)
```

### The model: all-MiniLM-L6-v2

`all-MiniLM-L6-v2` is a small, fast embedding model (~80MB) from the
sentence-transformers library. It outputs 384-dimensional vectors.

It is a distilled version of larger BERT models, trained specifically for
sentence-level semantic similarity tasks. It runs fast on CPU — embedding
a short phrase takes ~10ms.

### Why this specific model?

- **Small size (80MB)** — fits on any device
- **Fast on CPU** — no GPU needed
- **High quality for English** — top performer for sentence similarity benchmarks
- **Widely used** — well-tested, stable API

Used only during the `ingest` step (building the FAISS index). At runtime,
the stored index is loaded from disk — no embedding model needed for normal queries.

---

## 9. Silero VAD — voice activity detection

### What is VAD?

Voice Activity Detection is the task of determining whether a given audio segment
contains speech or silence/noise. Without VAD, you would not know when the customer
has finished speaking.

### What is Silero VAD?

Silero VAD is a tiny (~1MB) PyTorch neural network trained specifically for
speech/non-speech classification. It takes a 512-sample audio chunk (32ms at 16kHz)
and outputs a probability between 0 and 1:
- Close to 1.0 → speech detected
- Close to 0.0 → silence/noise

### The utterance detector state machine

```
State: WAITING
  → chunk arrives with speech prob > 0.5 → switch to SPEAKING
  → start accumulating chunks

State: SPEAKING
  → speech chunk → accumulate, reset silence counter
  → silence chunk → increment silence counter
  → silence counter reaches 26 chunks (≈800ms) → emit utterance, back to WAITING
```

The 800ms silence threshold is important: it avoids cutting off speech mid-sentence
(people naturally pause between phrases) while not waiting too long between customers.

### Why Silero over WebRTC VAD or energy-based VAD?

- **WebRTC VAD** is fast but rule-based (energy + spectral features). It produces
  many false positives with background noise common in pharmacy environments.
- **Energy-based** thresholding is even simpler and breaks completely in noisy rooms.
- **Silero VAD** is a learned neural network — it generalizes well to different
  noise conditions and accents, which matters in a multilingual pharmacy context.
- At ~1MB, it adds negligible memory overhead.

---

## 10. Faster-Whisper — speech to text

### What is Whisper?

Whisper is OpenAI's automatic speech recognition (ASR) model. It was trained on
680,000 hours of multilingual audio scraped from the internet. It can transcribe
and translate audio in 99 languages with high accuracy.

### What is faster-whisper?

`faster-whisper` is a re-implementation of Whisper using CTranslate2 — a fast
inference engine for transformer models. It runs 4x faster than the original
Whisper implementation and uses less memory.

This project uses the `small` model variant with `int8` quantization:
- **Small**: 465MB weights, good accuracy, handles accents and pharmacy terminology
- **int8**: quantized (weights compressed from float32 to int8), 2x smaller and faster at small accuracy cost

### Why not the base or tiny model?

`tiny` (75MB) and `base` (145MB) models frequently misrecognize medicine names
like "Dolo 650", "Crocin", "Glycomet" — these are not common English words and
require more model capacity to recognize correctly. The `small` model handles
them well.

### Language detection

Whisper auto-detects the spoken language and returns a confidence score.
If confidence is below 75%, this project defaults to English — preventing
incorrect translation of English pharmacy terms into wrong languages.

### Initial prompt bias

```python
initial_prompt = "Pharmacy kiosk. Medicines: Paracetamol, Dolo 650, Crocin..."
```

This hint tells Whisper to expect pharmacy-related vocabulary. It improves
recognition of brand names that would otherwise be mistranscribed (e.g.,
"Gly-co-met" instead of "Glycomet").

---

## 11. Piper TTS — text to speech

### What is Piper?

Piper is an offline, neural text-to-speech engine developed by the Home Assistant
project. It uses ONNX (Open Neural Network Exchange) format models and runs on CPU
without requiring a GPU.

ONNX is a standardized format for neural network models — it can run the same model
file on different hardware and software without retraining.

### Voice quality tiers

Piper voices come in four quality levels:
- `x_low` (~2MB): robot-like, very fast, works on Raspberry Pi
- `low` (~28MB): acceptable quality, fast
- `medium` (~60MB): natural-sounding, used for English and Hindi here
- `high` (100MB+): near-human quality, slower

### Language support

```
English  → en_US-lessac-medium  (American English, natural voice)
Hindi    → hi_IN-pratham-medium (Indian Hindi, natural voice)
Tamil    → ta_IN-x_low          (Tamil, lower quality — medium not available)
Urdu     → aliased to Hindi voice (similar phonetics)
```

### Lazy loading

Piper models are only loaded when first needed for a specific language.
An English-only session never loads the Hindi model. Once loaded, the model
is cached in memory for the rest of the session.

### Why not gTTS or Amazon Polly?

Both require internet. This kiosk must work offline. Piper produces
near-human quality voice on CPU with no network dependency.

---

## 12. argostranslate — offline translation

### What is argostranslate?

argostranslate is an offline machine translation library. It uses OpenNMT
(neural machine translation) models, packaged as downloadable language pairs.
Each language pair (e.g., English → Hindi) is ~30MB and is downloaded once,
then cached locally.

### Why translation is needed

Whisper transcribes the customer's Hindi/Tamil/Urdu speech. The LLM and all
tools operate in English. The final answer must be spoken back in the customer's
language. Translation bridges this gap.

### Current limitation

argostranslate quality for English → Hindi is acceptable but not perfect.
Pharmacy terms like "Paracetamol", "Dolo 650" are often left untranslated
by the model, which is actually correct behavior — brand names should not
be translated.

Only English → Hindi is actively supported. Tamil responses fall back to
English (Piper Tamil voice reads the English text).

### Why not Google Translate?

No internet dependency. The target users — rural pharmacy workers — may have
unreliable or no internet connectivity. All translation happens offline.

---

## 13. sounddevice — audio I/O

### What is sounddevice?

`sounddevice` is a Python library for capturing and playing audio using
PortAudio — a cross-platform audio I/O library. It works on Mac, Linux, and Windows
without hardware-specific code.

### Capture (streaming mode)

```python
sd.InputStream(
    samplerate=16000,
    channels=1,
    blocksize=512,       # 32ms chunks
    callback=_callback   # called every 32ms with new audio data
)
```

The callback runs in a separate audio thread. It puts each 512-sample chunk
into a queue. The main thread reads from the queue — this non-blocking design
prevents audio dropouts.

### Playback

```python
sd.play(audio_array, samplerate=22050, blocking=True)
```

`blocking=True` means the function returns only after the audio finishes playing.
This ensures the system does not start listening again while TTS is still speaking.

### The drain step

After TTS finishes, the mic buffer contains audio captured during playback
(the system's own voice feeding back into the mic, or room echo). `drain()`
discards this buffer so the VAD does not detect the TTS output as a new customer utterance.

---

## 14. Why Everything Runs Locally

This is the most important architectural decision in the project.

### The target environment

Rural and semi-urban pharmacies in India. Characteristics:
- **Unreliable internet** — 2G/3G, frequent outages
- **Low-end hardware** — basic Windows or Android devices
- **Multilingual customers** — Hindi, Tamil, Urdu, regional languages
- **Cost-sensitive** — cannot pay per API call

### Cloud API problems

| Problem | Impact |
|---|---|
| Network latency | 300-1000ms added per API call, making conversation feel sluggish |
| Offline failure | If internet drops, the kiosk goes completely dead |
| Cost | OpenAI API at $0.002/1K tokens × thousands of daily queries = significant cost |
| Data privacy | Customer medical queries should not leave the local device |

### Local stack tradeoffs

| Component | Local choice | Quality tradeoff |
|---|---|---|
| LLM | llama3.2:3b (2GB) | GPT-4 is much smarter but cloud-only |
| STT | Whisper small (465MB) | Slightly worse on strong accents vs Deepgram |
| TTS | Piper medium (60MB/voice) | Slightly robotic vs ElevenLabs |
| Translation | argostranslate (30MB) | Noticeably worse than Google Translate |

The tradeoffs are acceptable because the domain is narrow — pharmacy queries are
predictable and limited. The LLM does not need to know world history or write
poetry. It needs to reliably call ~7 tools based on ~50 common query patterns.

---

## 15. Architecture Decisions at a Glance

| Decision | Chosen approach | Why |
|---|---|---|
| Agent framework | LangGraph | Explicit loops, checkpointing, observable |
| LLM | llama3.2:3b via Ollama | Native tool calling, offline, free |
| No LLM on Round 2 | Direct Python formatting | Prevents small-model hallucination |
| Drug lookup | SQL keyword search | Fast, exact, no model loading |
| FAISS | Semantic fallback only | Handles synonym queries SQL misses |
| Cart state | In-memory (Python list) | Simple, reset per session, no persistence needed |
| Session memory | MemorySaver (RAM) | Multi-turn context, session-scoped |
| STT | faster-whisper small, int8 | CPU-fast, multilingual, handles brand names |
| VAD | Silero (neural) | Better than energy-based in noisy environments |
| TTS | Piper ONNX | Offline, near-human quality on CPU |
| Translation | argostranslate | Fully offline, one-time 30MB download |
| Audio I/O | sounddevice | Cross-platform, callback-based (no dropouts) |
| Inference server | Ollama (localhost) | OpenAI-compatible API, model management |

---

## Glossary

| Term | Meaning |
|---|---|
| **LLM** | Large Language Model — a neural network trained on text that can understand and generate language |
| **ReAct** | Reasoning + Acting — agent pattern that alternates between thinking and calling tools |
| **Tool calling** | LLM outputs a structured function call instead of prose text |
| **Embedding** | Converting text to a numerical vector that captures semantic meaning |
| **Vector search** | Finding the most similar vectors to a query vector (FAISS does this) |
| **VAD** | Voice Activity Detection — detecting when someone is speaking |
| **STT** | Speech-to-Text — converting audio to words |
| **TTS** | Text-to-Speech — converting words to audio |
| **ONNX** | Open Neural Network Exchange — portable format for neural network models |
| **int8 quantization** | Compressing model weights from 32-bit floats to 8-bit integers for speed/size |
| **Checkpointer** | Saves and restores graph state between invocations (enables conversation memory) |
| **Schedule H** | Indian pharmaceutical classification for prescription-only medicines |
| **Utterance** | A complete spoken phrase, from when the customer starts talking to when they stop |
| **Chunk** | A small fixed-size audio segment (512 samples = 32ms at 16kHz) |
