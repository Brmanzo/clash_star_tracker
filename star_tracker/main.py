#!/usr/bin/env python3
# main.py
# ------------------------------------------------------------
import cv2, os, pytesseract, sys
import FreeSimpleGUI as sg

from collections import OrderedDict
from pathlib import Path

from .data_structures import dataColumn, currentState
from .score_writeback import load_history, merge_new_war, rebuild_totals, write_history, print_leaderboard
from .image_measurement import menu_crop, measure_data_columns
from .image_processing import image_to_player_data
# ------------------------------------------------------------

state = currentState()  # Initialize the current state

# Set the path to the Tesseract executable
if not state.TESS_EXE:
    sys.exit("Tesseract executable not found. Please install Tesseract and ensure it is in your PATH.")
pytesseract.pytesseract.tesseract_cmd = state.TESS_EXE

def main() -> None:
    # --- popup returns a single *string* containing the paths separated by ';'
    files_str = sg.popup_get_file(
        'Select one or more screenshots',
        multiple_files=True,
        file_types=(('Image files', '*.png;*.jpg;*.jpeg'),),
        title='Open images for Clash Star Tracker')
    if not files_str:
        sg.popup('No files chosen')
        return                       # leave main()

    state.file_list = files_str.split(';')

    state.fileNum = 0

    for raw_path in state.file_list:
        state.fileNum += 1
        if not raw_path.lower().endswith(state.IMG_EXTS):
            continue
        state.image_path = Path(raw_path)
        state.debug_name = [state.image_path.stem,'.png']

        state.abs_pos, state.lineTop, state.nextLineTop, dataColumn.abs_pos = 0, 0, 0, 0

        state.src = cv2.imread(str(state.image_path))
        if state.src is None:
            print(f'Could not read {state.image_path}, skipping')
            continue

        if state.verbose: print(f"Processing {state.fileNum} of {len([f for f in state.file_list if f.lower().endswith(state.IMG_EXTS)])}: {raw_path}")

        # Refactored entire pipeline to these three functions
        state.attackLines = menu_crop(state)
        measure_data_columns(state)
        image_to_player_data(state)

        state.reset()

    # -------------------------------------------------- Calculate Final Score --------------------------------------------------
    print("\n--- Final War Data with Scores ---")
    for player in state.war_players:
        if player is not None:
            print(player.tabulate_player())
        else: continue
    
    state.new_scores = {}
    for player in state.war_players:
        if player is None or player.name is None:
            continue
        name  = player.name
        state.new_scores[name] = player.total_score()
    csv_path = os.path.join(state.PROJECT_ROOT, "../player_history.csv")

    # Load old history (or start fresh)
    try:
        _, history = load_history(csv_path)
    # If first ever run, create empty history
    except FileNotFoundError:
        history = OrderedDict()

    # Merge this war and update totals
    merge_new_war(history, state.new_scores)
    totals = rebuild_totals(history)

    # Write the full file back, still sorted by Total ↓
    print_leaderboard(history, totals)
    write_history(csv_path, history, totals)

if __name__ == "__main__":
    main()
