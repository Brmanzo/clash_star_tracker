# star_tracker/preprocessing.py
import cv2
import numpy as np
from typing import Tuple, List
from matplotlib import pyplot as plt

from .data_structures import dataColumn, currentState

def get_metrics(img_slice: np.ndarray) -> Tuple[float, float, float]:
    '''Helper to return the requested stat per slice of an image.'''
    if img_slice.size == 0: return 0.0, 0.0, 0.0
    # Tesseract works with uint8, so we convert to float for calculations
    float_slice = img_slice.astype(np.float32) / 255.0
    return float_slice.mean(), float_slice.min(), float_slice.max()

def measure_image(src: np.ndarray, bgThresh: float, behavior: str)-> Tuple[int, int]:
    '''Takes in an image, a given threshold, and a command to instruct behavior.
    
    Given an image of a single channel, and a threshold, the function returns the position
    of both events in the image.
    
    Behavior is specified through a string split by commas and allows for the following options:
        mode: 
            "absolute threshold": the threshold is meant to be taken from zero.
            "relative threshold": the threshold is meant to be taken from the previous value.
            "stat comparison": compares when the two chosen statistics diverge by a given threshold

        stat:
            "minimum", "maximum", or "average": to determine which statistic to compare with threshold
            In "stat comparison" mode, this is used to determine the first value to compare against the second.
                "min"/"max"/"average" < "min"/"max"/"average": to compare the first statistic against the second statistic.

        axis:
            "by row", or "by col": to indicate if iterating through image vertically or horizontally

        trig1:
            "first rise": the first trigger occurs when rising through threshold (prev is less than threshold, current is greater)
            "first fall": the first trigger occurs when falling through threshold (prev is greater than threshold, current is less)
            "from start": first event is not returned, and only returns the second event.
            "divergence": the first trigger occurs when the first statistic diverges from the second statistic by the threshold.

        when:
            "next": the second trigger occurs at the immediate next occurrence of the second trigger.
            "last": iterates to the end of the image, only remembering the final occurence of the second trigger.

        trig2 + Optional second condition:
            "rise", or "fall": indicates the behavior of the second trigger similar to the first trigger.
            "convergence": indicates the second trigger occurs when the first statistic converges with the second statistic by the threshold.
            
            Optional second condition can be indicated when more than one word is included in this argument.
                "while min", or "while max": specifies an additional condition for each trigger.
                "<", or ">": comparison operator relating stat to following threshold
                {float}: absolute threshold to be compared to for this second condition.
    '''
    # --- Step 1: Parse the behavior string ---
    parts = [p.strip() for p in behavior.split(',')]
    mode, stat, axis, trig1, when, trig2_full = parts
    
    # Parse the trigger and the optional secondary condition
    trig2_parts = trig2_full.split(" while ")
    trig2 = trig2_parts[0]
    extra_cond_str = trig2_parts[1] if len(trig2_parts) > 1 else None

    # --- Step 2: Initialization ---
    h, w = src.shape[:2]
    end = h if axis == "by row" else w
    m1, m2 = -1, -1

    if trig1 == "from start": m1 = 0
    if end < 2: return 0, end

    def get_metrics(img_slice):
        if img_slice.size == 0: return 0.0, 0.0, 0.0
        float_slice = img_slice.astype(np.float32) / 255.0
        return float_slice.mean(), float_slice.min(), float_slice.max()

    prev_mean, prev_min, prev_max = get_metrics(src[0:1, :] if axis == "by row" else src[:, 0:1])
    prevL = prev_mean if stat == "average" else (prev_min if stat == "minimum" else prev_max)

    # --- Step 3: Main Loop ---
    for i in range(1, end):
        L_slice = src[i:i+1, :] if axis == "by row" else src[:, i:i+1]
        curr_mean, curr_min, curr_max = get_metrics(L_slice)
        
        # This is the primary value we track for rise/fall
        currL = curr_mean if stat == "average" else (curr_min if stat == "minimum" else curr_max)
        
        # --- Unified Event Detection ---
        event_occurred = False
        if mode == "relative threshold":
            delta = currL - prevL
            if (trig1 == "first rise" or trig2 == "rise") and delta > bgThresh: event_occurred = True
            elif (trig1 == "first fall" or trig2 == "fall") and delta < -bgThresh: event_occurred = True
        elif mode == "absolute threshold":
            if (trig1 == "first rise" or trig2 == "rise") and currL >= bgThresh and prevL < bgThresh: event_occurred = True
            elif (trig1 == "first fall" or trig2 == "fall") and currL < bgThresh and prevL >= bgThresh: event_occurred = True
        elif mode == "stat comparison":
            # bgThresh is used to provide tolerance for the comparison
            rule_parts = stat.split(' ')
            val1 = (curr_min + bgThresh) if rule_parts[0] == 'min' else (curr_max - bgThresh)
            val2 = curr_mean if rule_parts[2] == 'average' else (curr_min + bgThresh if rule_parts[2] == 'min' else curr_max - bgThresh)
            op = rule_parts[1]
            
            if op == '<' and val1 < val2: event_occurred = True
            elif op == '>' and val1 > val2: event_occurred = True
        
        # --- Update Margins based on whether an event occurred ---
        if m1 == -1:
            if event_occurred and (trig1 == "first rise" or trig1 == "first fall" or trig1 == "divergence"):
                m1 = i
        else:
            if event_occurred and (trig2 == "rise" or trig2 == "fall" or trig2 == "convergence"):
                secondary_condition_met = True # Default to True
                if extra_cond_str:
                    # Parse the secondary condition
                    metric2, op2, thresh2_str = extra_cond_str.split(' ')
                    value_to_check = curr_min if metric2 == 'min' else curr_max
                    thresh2 = float(thresh2_str)
                    if op2 == '>': secondary_condition_met = value_to_check > thresh2
                    elif op2 == '<': secondary_condition_met = value_to_check < thresh2
                
                if secondary_condition_met:
                    if when == "last":
                        m2 = i
                    elif when == "next" and m2 == -1:
                        m2 = i
                        break
                            
        prevL = currL

    # --- Finalization ---
    if m1 == -1: m1 = 0
    if m2 == -1: m2 = end
    if m2 < m1: m2 = end

    return m1, m2

def sample_image(src: np.ndarray, behavior: str, globalTH: float|None, eps: float) -> float:
    '''Takes an image as input and a behavior command to sample the lightness of the image,
    given the requested statistic. A global threshold can be specified to help filter and 
    find a local maximum from a smaller window of the image. Epsilon is used to set the confidence
    for finding the first repeating value, which is returned as a threshold for further processing.
    
    Behavior is specified through a string split by commas and allows for the following options:
        type: 
                "max": return the greatest repeating value from the sampled statistic.
                "min": return the least repeating value from the sampled statistic.
                "avg": neutral placeholder to allow for default average sampling.

        mode:
                "absolute": the threshold is meant to be taken from zero.
                "relative": the threshold is meant to be taken from the previous value.

        stat:
                "minimum": samples the minimum lightness from the image.
                "maximum": samples the maximum lightness from the image.
                "average": samples the average lightness from the image.

        axis:
                "by row": samples the image by rows, iterating horizontally.
                "by col": samples the image by columns, iterating vertically.
        examples:
                "max, absolute, minimum, by row": returns the maximum minimum lightness by row.
                    Creates a helpful High-pass filter for only the greatest peaks in minimum lightness.
                "max, relative, average, by col": returns the minimum average lightness by column.
                    Creates a helpful High-pass filter for only the greatest changes in average lightness.
                "the, absolute, average, by row": returns the average lightness by row.
    '''

    # "max, absolute, minimum, by row"
    type, mode, stat, axis = behavior.split(", ")
    validType    = ['max', 'min', 'avg']
    validMode    = ['absolute', 'relative']
    validStat    = ['minimum', 'maximum', 'average']
    validAxis    = ['by row', 'by col']
    if type not in validType or mode not in validMode \
        or stat not in validStat or axis not in validAxis:
        raise ValueError("Invalid event parameter provided.")
    
    h, w = src.shape[:2]
    end = w if axis == "by row" else  h

    data, dataDx = [], []

    initial_metrics = get_metrics(src[0:1, :] if axis == "by row" else src[:, 0:1])
    prev = initial_metrics[0] if stat == "average" else (initial_metrics[1] if stat == "minimum" else initial_metrics[2])


    # Iterate through axis of image taking min, max, and avg for absolute threshold
    # As well as derivatives of each for relative thresholds
    for i in range(0, end):
        slice = src[i:i + 1,:] if axis == "by row" else  src[:, i:i + 1]
        if slice.size == 0: continue

        curr_mean, curr_min, curr_max = get_metrics(slice)
        curr = curr_mean if stat == "average" else (curr_min if stat == "minimum" else curr_max)

        data.append(curr)
        dataDx.append(curr - prev)
        prev = curr

    def first_repeat(data:List, globalTH: float|None) -> float:
        '''Finds the first repeating value, to be used as a filtering threshold '''
        i = 0
        if globalTH is not None:
             while data and abs(data[i] - globalTH) < eps: i += 1

        while i + 1 < len(data) and abs(data[i] - data[i + 1]) > eps:
            i += 1
        # if debug: 
        #     print(f"original data: {data[:10]}")
        #     print(f"local data: {data[i:i+10]}")
        return data[i]

    # Remove unique values from the front of list, leaving only the repeating values
    # If repeating Val is specified, global minimum is popped from list even if unique
    if mode == "relative":
        return first_repeat(sorted(dataDx, reverse=(type == "max")), globalTH)
    elif mode == "absolute":
        return first_repeat(sorted(data, reverse=(type == "max")), globalTH)
    else:  # 'avg'
        return sum(data) / end

def count_peaks(src: np.ndarray, thresh: float) -> int:
    '''Bootstraps measure_image function to count each occurence of a peak in input image.
    
    Behavior hardwired to absolute threshold, maximum, by col, from start, and returns the number of next rises.'''
    x = 0
    peaks = 0

    _, width = src.shape[:2]

    while x < width:
        _, next_x = measure_image(src[:, x:], thresh, behavior="absolute threshold, maximum, by col, from start, next, rise")
        if next_x == 0:
            break
        peaks += 1
        x += next_x
    return peaks

def debug_oscilloscope(s: currentState, dbgL: np.ndarray, graphName: str, plot_data: List[dataColumn]|None, axis: str) -> None:
    '''Oscilloscope-like function to plot lightness statistics over a given image for use in debugging'''
    
    lAvgData, lMinData, lMaxData = [], [], []
    h, w = dbgL.shape[:2]
    if axis == "row":
        dbgL = cv2.rotate(dbgL, cv2.ROTATE_90_COUNTERCLOCKWISE)
    for i in range(0, w):
        L = dbgL[:, i:i + 1]
        lAvgData.append(L.mean() / 255)
        lMaxData.append(L.max() / 255)
        lMinData.append(L.min() / 255)

    # Plotting with Image Background
    _, ax_image = plt.subplots(figsize=(20, 6))

    # Convert the single-channel grayscale image to a 3-channel BGR image
    background_img = cv2.cvtColor(dbgL, cv2.COLOR_GRAY2BGR)

    # Display the image on the primary axes (ax_image)
    h, w, _ = background_img.shape
    ax_image.imshow(background_img, aspect='auto', extent=(0, w, h, 0))
    ax_image.set_ylabel('Image Pixels (Height)', color='gray')
    ax_data = ax_image.twinx()

    # Plot lightness data
    ax_data.plot(lAvgData, label='Average Lightness', color='cyan', linewidth=2)
    ax_data.plot(lMinData, label='Minimum Lightness', color='#FF69B4', linewidth=2) # Hot Pink
    ax_data.plot(lMaxData, label='Maximum Lightness', color='#FFFF00', linewidth=2) # Yellow
    
    ax_data.set_ylabel('Lightness (0.0 - 1.0)', color='cyan')
    ax_data.tick_params(axis='y', labelcolor='cyan')
    ax_data.set_ylim(0, 1) # Lock the lightness axis from 0 to 1

    ax_image.set_xlabel("Column Index")
    plt.title("Lightness Data Plotted Over Image Slice")
    plt.grid(True, linestyle='--', alpha=0.5)

    out_path = str(s.OUT_DIR /f"{graphName}_lightness.png")
    plt.savefig(out_path)
    plt.close() # Close the plot to free up memory
    print(f"Saved combined plot to '{out_path}'")

    # Optional drawing of column boundaries when passed into second input
    if plot_data:
        for datum in plot_data:
            cv2.rectangle(dbgL, (datum.begin , 0), (datum.end, h - 1), (0, 0, 0), 2)

    out_ss_src = str(s.OUT_DIR /f"{graphName}_src_ss_.png")
    print(f"Saved original ss rect →", out_ss_src)
    cv2.imwrite(out_ss_src, dbgL)


def debug_image(s: currentState, image_to_save: np.ndarray, img_name: str) -> None:
    if image_to_save is None:
        print(f"Debug image {img_name} is None, skipping save.")
        return
    '''Outputs an image given a filename and current iterator value. '''
    if s.debug_name is None:
        print("Error: debug_name is not set in currentState. Cannot save debug image.")
        return
    out = str(s.OUT_DIR /f"{s.debug_name[0]}_{img_name}_{s.fileNum}.png")
    print(f"Saved preprocessed {img_name} →", out)
    cv2.imwrite(out, image_to_save)