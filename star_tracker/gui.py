# File: star_tracker/gui.py    
import cv2, json, os, sys, threading
import FreeSimpleGUI as sg
from typing import Optional, Tuple
from pathlib import Path
from collections import OrderedDict

from .data_structures import currentState, dataColumn
from .score_writeback import load_player_list
from .image_measurement import menu_crop, measure_data_columns
from .image_processing import image_to_player_data
from .score_writeback import load_history, merge_new_war, rebuild_totals, write_history, print_leaderboard

def load_settings(s: currentState) -> dict:
    """Loads settings from the JSON file. Returns an empty dict if not found."""
    try:
        with open(s.SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            print("Loaded previous settings.")
            return settings
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty/corrupt, return defaults
        print("No settings file found. Using defaults.")
        return {}

def save_settings(s: currentState, settings: dict):
    """Saves the given settings dictionary to the JSON file."""
    print("Saving settings...")
    with open(s.SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)

def run_backend_processing(s: currentState, window: sg.Window) -> None:
    status_elem = window['-STATUS-'] if window is not None and '-STATUS-' in window.AllKeysDict else None
    try:
        if not s.file_list:
            print("No images selected. Please choose images to process.")
            if status_elem is not None:
                status_elem.update(value="No images selected.", text_color='red')
            return
        for imagePath in s.file_list:
            if not str(imagePath).lower().endswith(s.IMG_EXTS):
                continue
            s.image_path = Path(imagePath)
            s.debug_name = [s.image_path.stem,'.png']
            s.abs_pos, s.lineTop, s.nextLineTop, dataColumn.abs_pos = 0, 0, 0, 0

            s.src = cv2.imread(str(s.image_path))
            if s.src is None:
                print(f'Could not read {s.image_path}, skipping')
                continue

            print(f"Processing {s.fileNum} of {len([f for f in s.file_list if f.name.lower().endswith(s.IMG_EXTS)])}: {imagePath}")

            # Refactored entire pipeline to these three functions
            s.attackLines = menu_crop(s)
            measure_data_columns(s)
            image_to_player_data(s)

            s.reset()

        print("\n--- Final War Data with Scores ---")
        for player in s.war_players:
            if player is not None:
                print(player.tabulate_player())
            else: continue
        
        s.new_scores = {}
        for player in s.war_players:
            if player is None or player.name is None:
                continue
            name  = player.name
            s.new_scores[name] = player.total_score()
        csv_path = os.path.join(s.PROJECT_ROOT, "../player_history.csv")

        # Load old history (or start fresh)
        try:
            history = load_history(csv_path)
        # If first ever run, create empty history
        except FileNotFoundError:
            history = OrderedDict()

        # Merge this war and update totals
        merge_new_war(history, s.new_scores)
        totals = rebuild_totals(history)

        # Write the full file back, still sorted by Total ↓
        print_leaderboard(history, totals)
        write_history(csv_path, history, totals)

    except Exception as e:
        print(f"\nFATAL ERROR DURING PROCESSING:\n{e}", file=sys.stderr)
        if 'status_elem' in locals() and status_elem is not None:
            status_elem.update(value="Error!", text_color='red')
    else:
        # This runs only if the try block completes WITHOUT an error
        if status_elem is not None:
            status_elem.update(value="Analysis Complete!", text_color='green')
    finally:
        run_button = window['-RUN-'] if window is not None and '-RUN-' in window.AllKeysDict else None
        if run_button is not None:
            run_button.update(disabled=False)



def run_gui(s: currentState) -> None:
    """Run the GUI for Clash Star Tracker."""
    sg.theme('light brown 3')

    settings = load_settings(s)

    # --- Step 1: Define the Window Layout ---

    # The Left Column contains only the Output element to display print statements.
    left_column = [
        [sg.Text("Program Output:")],
        [sg.Output(size=(80, 25), key='-OUTPUT-')]
    ]

    # The Right Column contains your settings and buttons.
    right_column = [
        [sg.Text("Configuration", font=("Helvetica", 16))],
        [sg.HSeparator()],
        [sg.Text("War Screenshots", size=(18, 1)), 
        sg.FilesBrowse("Choose Images...", key='-IMAGE_FILES-',
        file_types=(("Image Files", "*.png;*.jpg;*.jpeg"),), size=(28, 1))],

        # Open player list with viewable window and option to edit and save
        [sg.Text("Player List (.txt)", size=(18, 1)), 
         sg.Input( key='-PLAYERS_FILE-', size=(30, 1),
         default_text=settings.get('players_filepath', ''), enable_events=True),
         sg.FileBrowse(file_types=(("Text Files", "*.txt"),))],
        [sg.Multiline( default_text="Select a player file above to view/edit...",
         size=(45, 10), key='-PLAYER_LIST_TEXT-')],
        [sg.Button('Save Player List', key='-SAVE_PLAYERS-')],

        # Open multi-account file with viewable window
        [sg.Text("Multi-Account File (.json)", size=(18, 1)), 
         sg.Input(key='-MULTI_FILE-', size=(30, 1),
         default_text=settings.get('multi_filepath', '')),
         sg.FileBrowse(file_types=(("JSON Files", "*.json"),))],
        [sg.VPush()], # Pushes elements below it to the bottom
        [sg.Button("Run Analysis", key='-RUN-', size=(20, 2), button_color=('white', 'green'))],
        [sg.Text("Status: Ready", key='-STATUS-', text_color='green')],
        [sg.Button("Exit", size=(10, 1))]
    ]

    # If players.txt does not exist, create an empty file
    if not settings.get('players_filepath'):
        with open(s.PROJECT_ROOT / "players.txt", 'w', encoding='utf-8') as f:
            f.write("")

    # If multi.json does not exist, create an empty file
    if not settings.get('multi_filepath'):
        with open(s.PROJECT_ROOT / "multi_accounts.json", 'w', encoding='utf-8') as f:
            f.write("{}")

    # The final layout combines the two columns side-by-side.
    layout = [
        [sg.Column(left_column, element_justification='c'), 
         sg.Column(right_column, element_justification='c')]
    ]

    # --- Step 2: Create the Window ---
    window = sg.Window('Clash Star Tracker', layout, finalize=True)

    while True:
        read_result: Optional[Tuple[str, dict]] = window.read()
        if read_result is None or read_result[0] == sg.WIN_CLOSED:
            break
        event, values = read_result

        #  Save selected files when exiting
        if event == sg.WIN_CLOSED or event == 'Exit':
            settings_to_save = {
                'players_filepath': values['-PLAYERS_FILE-'],
                'multi_filepath': values['-MULTI_FILE-']
            }
            save_settings(s, settings_to_save)
            break  # Exit the loop

        if event == '-PLAYERS_FILE-':
            filepath = values['-PLAYERS_FILE-']
            
            # Make sure the path is valid and the file actually exists
            if filepath and os.path.exists(filepath):
                try:
                    # Read the entire content of the selected text file
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    # Use the .update() method to put the content into the Multiline element
                    player_list_elem = window['-PLAYER_LIST_TEXT-'] if '-PLAYER_LIST_TEXT-' in window.AllKeysDict else None
                    if player_list_elem is not None:
                        player_list_elem.update(value=text_content)
                        print(f"Successfully loaded and displayed: {filepath.split(os.sep)[-1]}")
                    else:
                        print("Error: Player list text element not found in the window.")
                except Exception as e:
                    sg.popup_error(f"Error reading file: {e}")
            
        # Saves modified player list to file
        if event == '-SAVE_PLAYERS-':
            filepath = values['-PLAYERS_FILE-']
            if filepath:
                try:
                    # Get the current text from the Multiline box
                    text_to_save = values['-PLAYER_LIST_TEXT-']
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(text_to_save)
                    sg.popup("Player list saved successfully!")
                    print(f"Saved changes to: {filepath}")
                except Exception as e:
                    sg.popup_error(f"Error saving file: {e}")
            else:
                sg.popup_error("Please select a player list file first before saving.")

        if event == '-RUN-':
            run_button = window['-RUN-']
            if run_button is None:
                print("Error: Run button not found in the window.")
                continue
            status_text = window['-STATUS-']
            if status_text is None:
                print("Error: Status text element not found in the window.")
                continue
            update_text = window['-OUTPUT-']
            if update_text is None:
                print("Error: Output text element not found in the window.")
                continue

            # Unpack event and values from read_result
            # --- Step 2: Validate Input Files ---
            players_filepath = values['-PLAYERS_FILE-']
            multi_filepath = values['-MULTI_FILE-']
            images_filepath = values['-IMAGE_FILES-']

            if not players_filepath or not multi_filepath or not images_filepath:
                sg.popup_error("Please specify both Player List, Multi-Account files, and War Screenshots.")
                continue


            # --- Step 3: Run and load the input data ---
            # Clear the output area
            print("\n--- Starting Clash Star Tracker Analysis ---\n")
            run_button.update(disabled=True)  # Disable the button to prevent multiple clicks
            status_text.update(value="Processing... please wait.", text_color='yellow')
            update_text.update(value='') # Clear the output window
            window.refresh()  # Refresh the window to show the changes

            if event == sg.WIN_CLOSED or event == 'Exit':
                # Create a dictionary with the current values to save
                settings_to_save = {
                    'players_filepath': values['-PLAYERS_FILE-'],
                    'multi_filepath': values['-MULTI_FILE-']
                }
                save_settings(s, settings_to_save)
                break # Then break the loop


            # Load player list and multi-account data
            s.players = load_player_list(players_filepath)
            with open(multi_filepath, encoding="utf-8") as f:
                s.multiAccounters = json.load(f)

            s.file_list = [Path(p) for p in images_filepath.split(';')]

            threading.Thread(
                target=run_backend_processing,
                args=(s, window),
                daemon=True
            ).start()

    window.close()
                