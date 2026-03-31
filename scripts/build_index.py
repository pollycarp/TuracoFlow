"""
Run this once to index all policy documents into LanceDB.

Usage:
    python scripts/build_index.py
"""

import sys
import logging
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from app.modules.rag import PolicyRAG


def main():
    print("Building TuracoFlow policy index...")
    rag = PolicyRAG()
    count = rag.build_index()
    print(f"\nDone. {count} chunks indexed into LanceDB.")
    print("You can now run the API: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
