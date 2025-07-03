# star_tracker/ocr.py

import cv2, re
import numpy as np
from fuzzywuzzy import process, utils
from typing import List

from .preprocessing import sample_image


def preprocess_line(img_bgr: np.ndarray, line:bool) -> np.ndarray:
    """Samples background of input image and returns a single channel preprocessed image
    where font color is black, and font outline and background are white. 
    
    Also returns the sampled highest minimum for adaptive thresholding. """

    BLOB_TH = 0.06     #  wipe blobs > 6 % of crop area
    OUTLINE_UPPER_BGR = (150, 150, 150)
    OUTLINE_LOWER_BGR = (0, 0, 0)

    # Thresholds for what lightness is considered the background for all background lightnesses
    LIGHT_ROW_TH = - 0.01 * 255
    DARK_UPPER_ROW_TH = 0.03 * 255
    DARK_LOWER_ROW_TH = 0.05 * 255
    USER_UPPER_ROW_TH = 0.09 * 255
    USER_LOWER_ROW_TH = 0.11 * 255
    
    h, w = img_bgr.shape[:2]

    # dark outline mask, all lightnesses between 0 and 150 are considered outline
    outline = cv2.inRange(img_bgr, OUTLINE_LOWER_BGR, OUTLINE_UPPER_BGR)

    # HLS -> lightness
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)[:, :, 1]

    # Estimate bg lightness from a small corner patch
    if line: x0 = 50; y0 = 20; x1 = 60; y1 = 30
    # If not line, more focused corner patch is required
    else: x0 = 0; y0 = 0; x1 = 5; y1 = 5        

    # estimate bg lightness from a small corner patch
    L_bg     = L[y0:y1, x0:x1].mean()
    L_bg_pct = L_bg/255

    # Background thresholding depends on sampled background lightness
    # Different cases for alternating line color and green user line
    match True:
        case _ if        L_bg_pct  >= 0.80: bg_thresh = L_bg + LIGHT_ROW_TH
        case _ if 0.77 <= L_bg_pct <  0.80: bg_thresh = L_bg + DARK_UPPER_ROW_TH
        case _ if 0.70 <= L_bg_pct <  0.77: bg_thresh = L_bg + DARK_LOWER_ROW_TH
        case _ if 0.62 <= L_bg_pct <  0.70: bg_thresh = L_bg + USER_UPPER_ROW_TH
        case _: bg_thresh                             = L_bg + USER_LOWER_ROW_TH

    # dark background (adaptive)
    _, dark_bg = cv2.threshold(L, bg_thresh, 255,
                               cv2.THRESH_BINARY_INV)

    # combined barrier for flood-fill
    barrier  = cv2.bitwise_or(outline, dark_bg)
    bg_fill  = barrier.copy()

    # flood border
    flood_b  = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(bg_fill, flood_b, (0, 0), 255)

    # NOR → keep only bright interior
    unwanted = cv2.bitwise_or(bg_fill, outline)
    keep     = cv2.bitwise_not(unwanted)
    glyphs   = cv2.bitwise_not(keep)

    # prune huge connected blobs of dark pixels
    max_blob           = int(BLOB_TH * h * w)
    inv                = cv2.bitwise_not(glyphs)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > max_blob:
            glyphs[lbl == i] = 255

    # Wipe a margin band of dark pixels around the image
    border = np.zeros_like(glyphs)
    border[:3, :], border[-3:, :] = 255, 255
    border[:, :3], border[:, -3:] = 255, 255
    ys, xs = np.where((border == 255) & (glyphs == 0))
    for y, x in zip(ys, xs):
        cv2.floodFill(glyphs, None, (int(x), int(y)), 255)

    return glyphs  # 0 = glyph ink, 255 = background

def auto_correct_num(num_OCR: str) -> int:
    '''When expecting a number but read a letter instead subsitute the character 
    for the commonly mistaken number in DIGIT_GLYPHS'''
    DIGIT_GLYPHS = "0-9lLiIoOsSzdeZWagTB|L"
    # Common subsitutions when reading rank
    TO_DIGIT = str.maketrans({'l':'1', 'I':'1',
                              '|':'1', 'L':'1',
                              'T':'1', 'g':'9',
                              'O':'0', 'o':'0',
                              'S':'5', 's':'5',
                              'B':'8', 'W':'11',
                              'Z':'2', 'z':'2',
                              'e':'2', 'a':'4',
                              'd':'1'})

    num_clean = re.sub(fr'[^{DIGIT_GLYPHS}]', '', num_OCR)
    digits = num_clean.translate(TO_DIGIT).strip(".")
    if not digits:
        return None          # or raise a clean exception
    return int(digits)

def auto_correct_player(player_OCR: str, confidence_threshold: int=65, enemy: bool=False, enemies: List[str]=[], players: List[str]=[]) -> str:
    '''Given a player name from OCR, match to an existing name from player table using fuzzy matching'''
    clean_name = utils.full_process(player_OCR)
    if clean_name and not enemy:
        best, score = process.extractOne(player_OCR, players)
    elif clean_name and enemy and enemies:
        best, score = process.extractOne(player_OCR, enemies)
    else:
        best, score = player_OCR.strip(), 0
    
    if best is None or score < confidence_threshold:
        best = player_OCR
        if best not in players and not enemy:
            players.append(best)
        else:
            enemies.append(best)
    return best

def score_from_stars(starsCentered: np.ndarray)-> str:
    '''Given a position in the star image, sample the lightness and determine if
    new, old, or no star.'''

    # Crop star by margins of 5 px to reduce error
    starsCentered = starsCentered[:, 5:-5]
    # min and max lightness do not change if old star, so if dx is zero, return old star
    no_star_TH = sample_image(starsCentered, "avg, relative, maximum, by col", None, eps=0.01)*0.99

    _, centeredWidth = starsCentered.shape[:2]
    LMin, LMax = [], []

    for i in range(0, centeredWidth):
        slice = starsCentered[:, i: i + 1]
        LMin.append(slice.min()/255)
        LMax.append(slice.max()/255)
    
    Max = max(LMax)

    # If 0.0, black present, and only new stars have black outline
    if Max == 1.00:           return "★"
    else:
        if no_star_TH == 0.0: return "_"
        else:                 return "☆"