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
from .score_writeback import load_history, merge_new_war, rebuild_totals, write_history

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

def save_settings(s: currentState, settings_to_save: dict):
    """Saves the given settings dictionary to the JSON file."""
    print("Saving settings...")
    with open(s.SETTINGS_FILE, 'w') as f:
        json.dump(settings_to_save, f, indent=4)

def print_to_gui(s: currentState, text_to_print: str):
    """A helper function to print text to the Multiline element."""
    # The '+=' appends the new text, and '\n' adds a newline.
    if s.window is None or '-OUTPUT-' not in s.window.AllKeysDict:
        return
    output_elem = s.window['-OUTPUT-'] if s.window is not None and '-OUTPUT-' in s.window.AllKeysDict else None
    if output_elem is not None:
        output_elem.update(value=text_to_print + '\n', append=True)
        s.window.refresh() # Force the GUI to update

def print_leaderboard(s: currentState, table: dict, totals: dict, width_name:int=22) -> None: # type: ignore
    '''Print "Rank  Name  Total" to the terminal.'''

    ordered = sorted(table.items(), key=lambda kv: (-totals[kv[0]], kv[0]))

    print_to_gui(s, "\n=== Current Leaderboard ===")
    for i, (player, _) in enumerate(ordered, start=1):
        total_score = totals[player]
        line = f"{i:>2}. {player:<{width_name}} Total Score: {total_score}"
        print_to_gui(s, line)

def run_backend_processing(s: currentState) -> None:
    '''Main processing pipeline to be threaded with GUI process'''
    status_elem = s.window['-STATUS-'] if s.window is not None and '-STATUS-' in s.window.AllKeysDict else None
    try:
        if not s.file_list:
            print_to_gui(s, "No images selected. Please choose images to process.")
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
                print_to_gui(s, f'Could not read {s.image_path}, skipping')
                continue

            print_to_gui(s, f"Processing {s.fileNum} of {len([f for f in s.file_list if f.name.lower().endswith(s.IMG_EXTS)])}: {imagePath}")

            # Refactored entire pipeline to these three functions
            s.attackLines = menu_crop(s)
            measure_data_columns(s)
            image_to_player_data(s)

            s.reset()

        print_to_gui(s, "\n--- Final War Data with Scores ---")
        s.new_scores = {}
        s.editable_lines = []
        s.editable_lines.append("Rank,Player Name,Final Score")

        for player in s.war_players:
            if player and player.name:
                player_line = player.tabulate_player()
                s.editable_lines.append(player_line)
                s.new_scores[player.name] = player.total_score()

        final_text = "\n".join(s.editable_lines)
        output_elem = s.window['-OUTPUT-'] if s.window is not None and '-OUTPUT-' in s.window.AllKeysDict else None
        if output_elem is not None:
            output_elem.update(value=final_text)

        if s.window is not None:
            s.window.metadata = {'new_scores': s.new_scores, 'csv_path': s.HISTORY_FILE}

        if status_elem is not None:
            status_elem.update(value="Review results and click Commit", text_color='cyan')
        commit_elem = s.window['-COMMIT-'] if s.window is not None and '-COMMIT-' in s.window.AllKeysDict else None
        if commit_elem is not None:
            commit_elem.update(disabled=False)

        for player in s.war_players:
            if player is not None:
                s.editable_lines.append(player.tabulate_player())
            else: continue
        output_elem = s.window['-OUTPUT-'] if s.window is not None and '-OUTPUT-' in s.window.AllKeysDict else None
        if output_elem is not None:
            output_elem.update(value=final_text)
        
    except Exception as e:
        print_to_gui(s, f"\nFATAL ERROR DURING PROCESSING:\n{e}")
        if 'status_elem' in locals() and status_elem is not None:
            status_elem.update(value="Error!", text_color='red')
    finally:
        run_button = s.window['-RUN-'] if s.window is not None and '-RUN-' in s.window.AllKeysDict else None
        if run_button is not None:
            run_button.update(disabled=False)



def run_gui(s: currentState) -> None:
    """Run the GUI for Clash Star Tracker."""
    sg.theme('light brown 3')

    s.settings = load_settings(s)

    # --- Step 1: Define the Window Layout ---

    # The Left Column contains only the Output element to display print statements.
    left_column = [
        [sg.Text("Program Output:")],
        [sg.Multiline("", size=(80, 25), key='-OUTPUT-',
         autoscroll=True, write_only=False)]]
    

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
         default_text=s.settings.get('players_filepath', ''), enable_events=True),
         sg.FileBrowse(file_types=(("Text Files", "*.txt"),))],
        [sg.Multiline( default_text="Select a player file above to view/edit...",
         size=(45, 10), key='-PLAYER_LIST_TEXT-')],
        [sg.Button('Save Player List', key='-SAVE_PLAYERS-')],

        # Open multi-account file
        [sg.Text("Multi-Account File (.json)", size=(18, 1)), 
         sg.Input(key='-MULTI_FILE-', size=(30, 1),
         default_text=s.settings.get('multi_filepath', ''), enable_events=True),
         sg.FileBrowse(file_types=(("JSON Files", "*.json"),))],
        [sg.Multiline( default_text="Select a player file above to view/edit...",
         size=(45, 10), key='-MULTI_LIST_TEXT-')],
        [sg.Button('Save Multi-Account List', key='-SAVE_MULTI_ACCOUNTS-')],

        # Open History file
        [sg.Text("Player History File (.csv)", size=(18, 1)), 
         sg.Input(key='-HISTORY_FILE-', size=(30, 1),
         default_text=s.settings.get('history_filepath', s.HISTORY_FILE)),
         sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],

        [sg.VPush()], # Pushes elements below it to the bottom
        [sg.Button("Run Analysis", key='-RUN-', size=(20, 2), button_color=('white', 'green')),
         sg.Button("Commit to History", key='-COMMIT-', size=(20, 2),
         disabled=True)],
        [sg.Text("Status: Ready", key='-STATUS-', text_color='green')],
        [sg.Button("Exit", size=(10, 1))]
    ]

    # If players.txt does not exist, create an empty file
    players_path_str = s.settings.get('players_filepath')
    if not players_path_str or not Path(players_path_str).is_file():
        # Define the default path within your project
        default_path = s.PROJECT_ROOT / "players.txt"
        print(f"Player list not found or path is invalid. Creating default at: {default_path}") 
        default_path.touch()
        s.settings['players_filepath'] = str(default_path)
    # If multi_accounts.json does not exist, create an empty file
    multi_path_str = s.settings.get('multi_filepath')
    if not multi_path_str or not Path(multi_path_str).is_file():
        default_path = s.PROJECT_ROOT / "multi_accounts.json"
        print(f"Multi-account file not found or path is invalid. Creating default at: {default_path}")
        default_path.write_text("{}", encoding="utf-8")
        s.settings['multi_filepath'] = str(default_path)

    history_path_str = s.settings.get('history_filepath')
    if not history_path_str or not Path(history_path_str).is_file():
        default_path = s.PROJECT_ROOT / "player_history.csv"
        print(f"Player history file not found or path is invalid. Creating default at: {default_path}")
        default_path.touch()
        s.settings['history_filepath'] = str(default_path)


    # The final layout combines the two columns side-by-side.
    layout = [
        [sg.Column(left_column, element_justification='c'), 
         sg.Column(right_column, element_justification='c')]
    ]

    # --- Step 2: Create the Window ---
    s.window = sg.Window('Clash Star Tracker', layout, finalize=True)

    # ------------------------------------- Main Event Loop -------------------------------------
    while True:
        read_result: Optional[Tuple[str, dict]] = s.window.read()
        if read_result is None or read_result[0] == sg.WIN_CLOSED:
            break
        event, values = read_result

        #  Save selected files when exiting
        if event == sg.WIN_CLOSED or event == 'Exit':
            settings_to_save = {
                'players_filepath': values['-PLAYERS_FILE-'],
                'multi_filepath': values['-MULTI_FILE-'],
                'history_filepath': values['-HISTORY_FILE-']
            }
            save_settings(s, settings_to_save)
            break  # Exit the loop
        # --------------------------------------- Handle Events ---------------------------------------
        if event == '-PLAYERS_FILE-':
            filepath = values['-PLAYERS_FILE-']
            
            # Make sure the path is valid and the file actually exists
            if filepath and os.path.exists(filepath):
                try:
                    # Read the entire content of the selected text file
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    # Use the .update() method to put the content into the Multiline element
                    player_list_elem = s.window['-PLAYER_LIST_TEXT-'] if '-PLAYER_LIST_TEXT-' in s.window.AllKeysDict else None
                    if player_list_elem is not None:
                        player_list_elem.update(value=text_content)
                        print(f"Successfully loaded and displayed: {filepath.split(os.sep)[-1]}")
                    else:
                        print("Error: Player list text element not found in the window.")
                except Exception as e:
                    sg.popup_error(f"Error reading file: {e}")
            
        # Saves modified player list to file
        elif event == '-SAVE_PLAYERS-':
            filepath = values['-PLAYERS_FILE-']
            if filepath and os.path.exists(filepath):
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

        elif event == '-MULTI_FILE-':
            filepath = values['-MULTI_FILE-']
            
            # Make sure the path is valid and the file actually exists
            if filepath and os.path.exists(filepath):
                try:
                    # Read the entire content of the selected JSON file
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    # Use the .update() method to put the content into the Multiline element
                    multi_list_elem = s.window['-MULTI_LIST_TEXT-'] if '-MULTI_LIST_TEXT-' in s.window.AllKeysDict else None
                    if multi_list_elem is not None:
                        multi_list_elem.update(value=text_content)
                        print(f"Successfully loaded and displayed: {filepath.split(os.sep)[-1]}")
                    else:
                        print("Error: Multi-account list text element not found in the window.")
                except Exception as e:
                    sg.popup_error(f"Error reading file: {e}")

        elif event == '-RUN-':
            run_button = s.window['-RUN-']
            if run_button is None:
                print("Error: Run button not found in the window.")
                continue
            status_button = s.window['-STATUS-']
            if status_button is None:
                print("Error: Status button element not found in the window.")
                continue
            update_text = s.window['-OUTPUT-']
            if update_text is None:
                print("Error: Output text element not found in the window.")
                continue
            commit_button = s.window['-COMMIT-']
            if commit_button is None:
                print("Error: Commit button not found in the window.")
                continue

            # --- Step 3: Run and load the input data ---
            # Clear the output area
            print("\n--- Starting Clash Star Tracker Analysis ---\n")
            run_button.update(disabled=True)  # Disable the button to prevent multiple clicks
            status_button.update(value="Processing... please wait.", text_color='yellow')
            commit_button.update(disabled=True)  # Disable commit button until processing is done
            s.window.refresh()  # Refresh the window to show the changes

            # Unpack event and values from read_result
            # --- Step 2: Validate Input Files ---
            players_filepath = values['-PLAYERS_FILE-']
            multi_filepath = values['-MULTI_FILE-']
            images_filepath = values['-IMAGE_FILES-']

            if not players_filepath or not multi_filepath or not images_filepath:
                sg.popup_error("Please specify both Player List, Multi-Account files, and War Screenshots.")
                continue

                        # Load player list and multi-account data
            s.players = load_player_list(players_filepath)
            with open(multi_filepath, encoding="utf-8") as f:
                s.multiAccounters = json.load(f)

            s.file_list = [Path(p) for p in images_filepath.split(';')]

            threading.Thread(
                target=run_backend_processing,
                args=(s,),
                daemon=True
            ).start()

        elif event == sg.WIN_CLOSED or event == 'Exit':
            # Create a dictionary with the current values to save
            settings_to_save = {
                'players_filepath': values['-PLAYERS_FILE-'],
                'multi_filepath': values['-MULTI_FILE-']
            }
            save_settings(s, settings_to_save)
            break # Then break the loop

        elif event == '-COMMIT-':
            edited_text = values['-OUTPUT-']

            new_scores_from_edit = {}
            lines = edited_text.strip().split('\n')
            for line in lines[1:]:
                try:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        name = parts[1].strip()
                        new_score = parts[-1].strip()
                        new_scores_from_edit[name] = new_score
                except Exception as e:
                    sg.popup_error(f"Error parsing edited text: {line}\n{e}")

            try:
                print("Committing new war data to history...")
     
                # Load old history (or start fresh)
                try:
                    history = load_history(s.HISTORY_FILE)
                except FileNotFoundError:
                    history = OrderedDict()
                
                # Merge this war and update totals
                merge_new_war(history, new_scores_from_edit)
                totals = rebuild_totals(history)

                write_history(s.HISTORY_FILE, history, totals)
                s.window.metadata = {'history': history, 'totals': totals, 'csv_path': s.HISTORY_FILE}

                sg.popup("History committed successfully!")
                print_leaderboard(s, history, totals)
                # Update the status and enable the commit button
                status_elem = s.window['-STATUS-'] if s.window is not None and '-STATUS-' in s.window.AllKeysDict else None
                if status_elem is not None:
                    status_elem.update(value="History Saved. Ready.", text_color='green')
                commit_elem = s.window['-COMMIT-'] if s.window is not None and '-COMMIT-' in s.window.AllKeysDict else None
                if commit_elem is not None:
                    commit_elem.update(disabled=True)
            except Exception as e:
                sg.popup_error(f"Error committing history: {e}")
                if status_elem is not None:
                    status_elem.update(value="Error saving history!", text_color='red')


    s.window.close()
