# Example End-to-End Interaction Flow

## User Query

**Customer says:**  
> "I have a headache, do you have anything?"

---

# Voice Pharmacy Assistant — Execution Pipeline

## 1) Audio Capture
**Component:** `Mic → AudioCapture`

- Microphone continuously streams raw audio.
- Audio is collected in **512-sample chunks**.
- These chunks are buffered and passed downstream for speech detection.

**Output:**  
Raw audio stream

---

## 2) Voice Activity Detection (VAD)
**Component:** `SileroVAD`

- The system checks whether the incoming audio contains speech.
- While the user is speaking, audio is continuously accumulated.
- Once **~800 ms of silence** is detected, the utterance is considered complete.

### Behavior
- Start buffering when speech begins
- Continue accumulating while speech is active
- Stop buffering when silence threshold is crossed
- Emit one complete speech segment for transcription

**Output:**  
Final utterance audio segment

---

## 3) Speech-to-Text (STT)
**Component:** `WhisperSTT`

The buffered speech segment is transcribed into text.

### Example Output
```python
("I have a headache, do you have anything?", "en")
```

Where:
- `"I have a headache, do you have anything?"` = transcript
- `"en"` = detected language

**Output:**  
Transcribed user query + language

---

## 4) Input to Agent Graph
**Component:** `process_text()`

The transcript is wrapped into a conversational message object and passed into the agent pipeline.

### Example
```python
HumanMessage(content="I have a headache, do you have anything?")
```

This message is inserted into the graph / state machine.

**Output:**  
Structured user message in agent state

---

## 5) LLM Reasoning — Round 1
**Component:** `call_model()`

The LLM receives the user message and reasons about the intent.

### Detected Intent
The user is asking for a medicine recommendation based on a symptom:
- Symptom detected: **headache**

### LLM Decision
Instead of answering directly, the model decides it needs inventory / medicine lookup.

### Tool Call Generated
```python
drug_lookup("headache")
```

**Output:**  
Tool invocation request

---

## 6) Tool Argument Normalization
**Component:** `_normalize_args`

Before tool execution, the arguments are cleaned and standardized.

### Example
```python
input: "headache"
output: "headache"
```

This step helps prevent malformed tool calls and ensures consistency.

**Output:**  
Validated tool arguments

---

## 7) Inventory Search / Drug Lookup
**Component:** `InventoryManager.drug_lookup("headache")`

The tool searches the medicine database for relevant drugs.

### Internal Logic

#### Keyword Extraction
After stopword filtering:

```python
keywords = ["headache"]
```

#### Search Strategy
The inventory database is queried using symptom/use matching.

```sql
WHERE uses LIKE "%headache%"
```

### Matching Medicines Found

#### Generic Drug
- **Paracetamol**
  - Category: **Analgesic**
  - Uses: **Headache, fever, pain**

#### Available Brands in Stock
- **Dolo 650** *(Best Seller)*
- **Crocin 500**

### Tool Result Example
```python
{
  "generic_name": "Paracetamol",
  "category": "Analgesic",
  "uses": "Headache, fever, pain",
  "brands": ["Dolo 650", "Crocin 500"]
}
```

**Output:**  
Structured medicine recommendation data

---

## 8) Tool Result Added to State
**Component:** `ToolMessage`

The tool output is inserted back into the conversation state so the LLM can use it to generate a final user-facing response.

**Output:**  
Updated graph state with tool result

---

## 9) LLM Response Generation — Round 2
**Component:** `call_model()`

The LLM now receives:
- Original user query
- Tool output from `drug_lookup()`

It formats the final conversational response.

### Formatting Step
```python
_format_direct("drug_lookup", result)
```

### Generated Response
> "For that I recommend **Paracetamol**. We have **Dolo 650** and **Crocin 500** in stock. Would you like to add one?"

**Output:**  
Natural language assistant reply

---

## 10) Translation Layer (Optional)
**Component:** Response post-processing

- Since the user spoke in **English** and the reply is already in English:
  - **No translation is needed**

**Output:**  
Final text response in user language

---

## 11) Text-to-Speech (TTS)
**Component:** `PiperTTS (en)`

The assistant response is converted into spoken audio.

### Input
```text
"For that I recommend Paracetamol. We have Dolo 650 and Crocin 500 in stock. Would you like to add one?"
```

### Output
- Synthesized speech waveform

This is then played back to the user.

### Playback
```python
sounddevice.play(...)
```

**Output:**  
Spoken assistant reply

---

## 12) Microphone Buffer Cleanup
**Component:** `AudioCapture.drain()`

After speaking back, the microphone input buffer is flushed to avoid:
- TTS audio leaking back into STT
- stale audio chunks being reprocessed
- feedback loops

**Output:**  
Clean audio state

---

## 13) Return to Listening Mode

The system goes back to idle listening and waits for the next user utterance.

**System State:**  
Ready for next interaction

---

# Compact Flow Summary

```text
Customer speaks
   ↓
AudioCapture
   ↓
SileroVAD detects speech
   ↓
Silence > 800ms → emit utterance
   ↓
WhisperSTT transcribes audio
   ↓
process_text() → HumanMessage
   ↓
LLM Round 1 reasons about intent
   ↓
Tool call: drug_lookup("headache")
   ↓
InventoryManager searches database
   ↓
Returns Paracetamol → Dolo 650 / Crocin 500
   ↓
ToolMessage added to state
   ↓
LLM Round 2 generates final answer
   ↓
PiperTTS speaks response
   ↓
Audio buffer drained
   ↓
System returns to listening
```

---

# One-Line System View

## Architecture
```text
Mic → VAD → STT → LLM Agent → Tool/Inventory Lookup → LLM Response → TTS → Speaker
```

## With State
```text
Mic → VAD → Whisper → Agent Graph ↔ Tools/Inventory ↔ Conversation Memory → Piper → Speaker
```

---

# Why This Flow Works Well

This architecture is strong because it is:

- **Realtime enough** for natural conversation
- **Tool-grounded** instead of hallucinating medicine suggestions
- **Inventory-aware** (recommends only available stock)
- **Language-aware** (can support multilingual later)
- **Modular** (easy to swap Whisper, Piper, VAD, LLM, DB, etc.)

---

# Example Final Assistant Reply

> "For that I recommend **Paracetamol**. We have **Dolo 650** and **Crocin 500** in stock. Would you like to add one?"