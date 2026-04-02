"""
main.py — Entry point for MedAssistant.

Usage:
  python main.py              # full voice mode (mic + speakers)
  python main.py --text       # text mode (keyboard input, no audio hardware)
  python main.py --ingest     # re-build the drug database index
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="MedAssistant — Voice-first medicine assistant for rural healthcare workers"
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Run in text mode (keyboard input, no mic/speakers needed)"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Re-build the FAISS drug database index from medicines.csv"
    )
    args = parser.parse_args()

    if args.ingest:
        from rag.ingest import ingest
        ingest()
        return

    from pipeline import run
    run(text_only=args.text)


if __name__ == "__main__":
    main()
