#!/usr/bin/env python3
# star_tracker/main.py
# ------------------------------------------------------------
import pytesseract, sys

from .state import currentState
from .gui import run_gui
# ------------------------------------------------------------

state = currentState()  # Initialize the current state

# Set the path to the Tesseract executable
if not state.TESS_EXE:
    sys.exit("Tesseract executable not found. Please install Tesseract and ensure it is in your PATH.")
pytesseract.pytesseract.tesseract_cmd = state.TESS_EXE

def main() -> None:
    run_gui(state)

if __name__ == "__main__":
    main()
