"""
audio/translator.py

Offline translation using argostranslate (OpenNMT-based, ~30MB per language pair).

Why a separate translator instead of asking the LLM to reply in Hindi?
  phi3:mini (3.8B, English-primary) generates broken Hindi — mixing English
  words written in Devanagari script. A dedicated 30MB translation model
  trained specifically for en→hi/ta produces correct text.

Flow:
  LLM always responds in English (what it does well)
       ↓
  Translator converts to target language (what it does well)
       ↓
  Piper TTS speaks in target language

Language packages (~30MB each) are downloaded once on first use, then cached.
Fully offline after first download.
"""

from __future__ import annotations

from argostranslate import package as at_package, translate as at_translate

# Languages argostranslate can translate TO from English (verified against package index)
# hi ✓, ur ✓  |  ta ✗ (no package), te ✗, mr ✗, bn ✗ (no Piper voice)
SUPPORTED_TARGETS = {"hi", "ur"}

# For Urdu: translate to Hindi text (mutually intelligible) + use Hindi Piper voice
# This works because Devanagari Piper models can't speak Arabic/Nastaliq script
TRANSLATE_AS: dict[str, str] = {
    "ur": "hi",   # translate to Hindi, spoken with Hindi voice
    "mr": "hi",   # Marathi → Hindi (closest available)
    "ne": "hi",   # Nepali → Hindi
}

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "ur": "Urdu",
    "bn": "Bengali",
    "mr": "Marathi",
}

_installed: set[str] = set()   # track which en→X packages are ready


def _ensure_package(target: str) -> bool:
    """
    Download and install the en→target language package if not already present.
    Returns True if ready, False if installation failed.
    """
    if target in _installed:
        return True

    try:
        # Check what's already installed locally
        installed_langs = at_translate.get_installed_languages()
        codes = {lang.code for lang in installed_langs}

        if "en" in codes and target in codes:
            _installed.add(target)
            return True

        # Need to download
        print(f"[Translator] Downloading en→{target} package (~30MB, one-time)...")
        at_package.update_package_index()
        available = at_package.get_available_packages()

        pkg = next(
            (p for p in available if p.from_code == "en" and p.to_code == target),
            None,
        )
        if pkg is None:
            print(f"[Translator] No package found for en→{target}")
            return False

        at_package.install_from_path(pkg.download())
        print(f"[Translator] en→{target} ready.")
        _installed.add(target)
        return True

    except Exception as e:
        print(f"[Translator] Failed to set up en→{target}: {e}")
        return False


def translate(text: str, target_lang: str) -> str:
    """
    Translate English text to target_lang.
    Returns original English text if translation fails or target is English.

    target_lang: ISO 639-1 code ("hi", "ta", "ur", "mr", ...)
    """
    if not text.strip():
        return text

    if target_lang == "en":
        return text

    # Remap to what we can actually translate to
    target_lang = TRANSLATE_AS.get(target_lang, target_lang)

    if target_lang not in SUPPORTED_TARGETS:
        # No translation available — return English (pipeline will use English TTS)
        return text

    if not _ensure_package(target_lang):
        print(f"[Translator] Falling back to English (package unavailable)")
        return text

    try:
        installed = at_translate.get_installed_languages()
        en_lang = next((l for l in installed if l.code == "en"), None)
        tgt_lang = next((l for l in installed if l.code == target_lang), None)

        if en_lang is None or tgt_lang is None:
            return text

        translation = en_lang.get_translation(tgt_lang)
        result = translation.translate(text)
        print(f"[Translator] {LANG_NAMES.get(target_lang, target_lang)}: {result}")
        return result

    except Exception as e:
        print(f"[Translator] Translation error: {e}. Using English.")
        return text
