# star_tracker/ocr.py

import cv2, numpy as np, re, sys
from fuzzywuzzy import process, utils

from .preprocessing import sample_image
from .state import currentState, print_to_gui


def preprocess_line(s: currentState, img_bgr: np.ndarray, line:bool) -> np.ndarray:
    """Samples background of input image and returns a single channel preprocessed image
    where font color is black, and font outline and background are white. 
    
    Also returns the sampled highest minimum for adaptive thresholding. """

    # Thresholds for what lightness is considered the background for all background lightnesses
    
    h, w = img_bgr.shape[:2]

    # dark outline mask, all lightnesses between 0 and 150 are considered outline
    outline = cv2.inRange(img_bgr, s.presets.OUTLINE_LOWER_BGR, s.presets.OUTLINE_UPPER_BGR)

    # HLS -> lightness
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)[:, :, 1]

    # Estimate bg lightness from a small corner patch
    if line: x0, y0, x1, y1 = s.presets.lineBgSampling[:]
    # If not line, more focused corner patch is required
    else: x0, y0, x1, y1 = s.presets.cornerBgSampling[:]

    # estimate bg lightness from a small corner patch
    L_bg     = L[y0:y1, x0:x1].mean()
    L_bg_pct = L_bg/255

    # Background thresholding depends on sampled background lightness
    # Different cases for alternating line color and green user line
    bg_thresh = L_bg + s.presets.lowerUserTH.delta

    for threshold_preset in s.presets.thresholdMap:
        # Because the list is sorted from low to high, this will find the tightest, correct bound
        if L_bg_pct >= threshold_preset.bound:
            bg_thresh = L_bg + threshold_preset.delta

    # dark background (adaptive)
    _, dark_bg = cv2.threshold(L, bg_thresh, 255,
                               cv2.THRESH_BINARY_INV)

    # combined barrier for flood-fill
    barrier  = cv2.bitwise_or(outline, dark_bg)
    bg_fill  = barrier.copy().astype(np.uint8)

    # flood border
    mask  = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(bg_fill, mask, (0, 0), (255,))

    # NOR → keep only bright interior
    unwanted = cv2.bitwise_or(bg_fill, outline)
    keep     = cv2.bitwise_not(unwanted)
    glyphs   = cv2.bitwise_not(keep)

    # prune huge connected blobs of dark pixels
    max_blob           = int(s.presets.BLOB_TH * h * w)
    inv                = cv2.bitwise_not(glyphs)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > max_blob:
            glyphs[lbl == i] = 255

    # Wipe a margin band of dark pixels around the image
    border = np.zeros_like(glyphs)
    border[:3, :], border[-3:, :] = 255, 255
    border[:, :3], border[:, -3:] = 255, 255
    ys, xs = np.where((border == 255) & (glyphs == 0))
    for y, x in zip(ys, xs):
        cv2.floodFill(glyphs, None, (int(x), int(y)), (255,))

    return glyphs  # 0 = glyph ink, 255 = background

def auto_correct_num(s: currentState, num_OCR: str) -> int|None:
    '''When expecting a number but read a letter instead subsitute the character 
    for the commonly mistaken number in DIGIT_GLYPHS'''
    num_clean = re.sub(fr'[^{s.presets.DIGIT_GLYPHS}]', '', num_OCR)
    digits = num_clean.translate(s.presets.TO_DIGIT).strip(".")
    if not digits:
        return None          # or raise a clean exception
    return int(digits)

def auto_correct_player(s: currentState, player_OCR: str, confidence_threshold: int=65, enemy: bool=False) -> str:
    '''Given a player name from OCR, match to an existing name from player table using fuzzy matching'''
    clean_name = utils.full_process(player_OCR)
    if s.players is None or s.enemies is None:
        print_to_gui(s, f"Error: players or enemies list is None for image {s.fileNum}. Exiting.")
        sys.exit(1)
    if clean_name and not enemy:
        result = process.extractOne(player_OCR, s.players)
        if result is not None:
            best, score = result[:2]
        else:
            best, score = player_OCR.strip(), 0
    elif clean_name and enemy and s.enemies:
        result = process.extractOne(player_OCR, s.enemies)
        if result is not None:
            best, score = result[:2]
        else:
            best, score = player_OCR.strip(), 0
    else:
        best, score = player_OCR.strip(), 0
    
    if best is None or score < confidence_threshold:
        best = player_OCR
        if best not in s.players and not enemy:
            s.players.append(best)
        else:
            s.enemies.append(best)
    return best

def score_from_stars(s: currentState, starsCentered: np.ndarray)-> str:
    '''Given a position in the star image, sample the lightness and determine if
    new, old, or no star.'''

    # Crop star by margins of 5 px to reduce error
    starsCentered = starsCentered[:, s.presets.STAR_MARGIN:-s.presets.STAR_MARGIN]
    # min and max lightness do not change if old star, so if dx is zero, return old star
    no_star_TH = sample_image(starsCentered, "avg, relative, maximum, by col",
                              None, s.presets.no_star_TH.repCharTol)*s.presets.no_star_TH.filterScale

    _, centeredWidth = starsCentered.shape[:2]
    LMin, LMax = [], []

    for i in range(0, centeredWidth):
        slice = starsCentered[:, i: i + 1]
        LMin.append(slice.min()/255)
        LMax.append(slice.max()/255)
    
    Max = max(LMax)

    # If 0.0, black present, and only new stars have black outline
    if Max == 1.00:           return "☆"
    else:
        if no_star_TH == 0.0: return "_"
        else:                 return "★"