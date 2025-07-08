import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_structures import dataColumn

class sampleImagePresets:
    def __init__(self, repCharTol: float, filterScale: float):
        self.repCharTol = repCharTol
        self.filterScale = filterScale

class ImageMeasurements:
    """Holds the image measurement presets for sampling."""
    
    def __init__(self):
        self.menuTopMargin: int|None        = None
        self.menuBottomMargin: int|None     = None
        self.menuLeftMargin: int|None       = None
        self.menuRightMargin: int|None      = None
        
        self.headerEnd: int|None            = None
        self.lineBegning: int|None          = None
        self.lineEnd: int|None              = None

        self.rankCol: dataColumn|None       = None
        self.levelCol: dataColumn|None      = None
        self.playerCol: dataColumn|None     = None
        self.enemyCol: dataColumn|None      = None
        self.percentageCol: dataColumn|None = None
        self.starsCol: dataColumn|None      = None

class backgroundThresholds:
    def __init__(self, bound: float, delta: float):
        self.bound = bound
        self.delta = delta*255 # Convert delta to pixel value


def _float_or_default(settings: dict, key: str, default: float) -> float:
    """Helper to get a float from settings dict, or return default if not present or not convertible."""
    try:
        return float(settings.get(key, default))
    except (TypeError, ValueError):
        return default

class processingPresets:
    """Presets for image sampling tolerances, can be tweaked in Advanced Settings."""
    
    def update_from_dict(self, settings: dict):
        """
        Updates the class attributes from a loaded settings dictionary.
        Uses .get(key, default_value) to safely access the dictionary.
        """
        print("Updating presets from loaded settings...")

        
        # For your sampleImagePresets objects, we update them individually
        # This uses .get() to avoid an error if the key is missing from the JSON

        sampling_advanced_settings = {
            #  attr                      json-key-epsilon             json-key-scale
            "col_src_avg_TH"      : ("Horizontal Background Crop Epsilon",
                                     "Horizontal Background Crop Scale Factor"),
            "row_src_avg_TH"      : ("Vertical Background Crop Epsilon",
                                     "Vertical Background Crop Scale Factor"),
            "col_menu_max_avg_TH" : ("Horizontal Menu Crop Epsilon",
                                     "Horizontal Menu Crop Scale Factor"),
            "row_menu_min_TH"     : ("Vertical Menu Crop Epsilon",
                                     "Vertical Menu Crop Scale Factor"),
            "col_al_local_min_TH" : ("Horizontal Lines Local Minimum Epsilon",
                                     "Horizontal Lines Local Minimum Scale Factor"),
            "col_al_global_min_TH": ("Horizontal Lines Global Minimum Epsilon",
                                     "Horizontal Lines Global Minimum Scale Factor"),
            "col_al_sep_TH"       : ("Horizontal Data Column Separation Epsilon",
                                     "Horizontal Data Column Separation Scale Factor"),
            "text_menu_TH"        : ("Rank-Name Separation Epsilon",  
                                     "Rank-Name Separation Scale Factor"),
            "preproc_attack_avgL" : ("Empty Attack Line Epsilon",
                                     "Empty Attack Line Scale Factor"),
            "new_line_TH"         : ("New line separation Epsilon",
                                     "New line separation Scale Factor"),
            "no_star_TH"          : ("Old Star Noise Epsilon",
                                     "Old Star Noise Scale Factor")
        }
        preprocessing_advanced_settings = {
            #  attr                      json-key-key
            "lightnessUpperBound" : "Preprocessing Light Upperbound",
            "lightnessLowerBound" : "Preprocessing Light Lowerbound",
            "BLOB_TH"             : "Blob to Remove Size Percentage",
            "lineBgSampling"      : "Line Background Sampling (x0, y0, x1, y1)",
            "cornerBgSampling"    : "Small Corner Background Sampling (x0, y0, x1, y1)"
        }

        preprocessing_background_threshold_settings = {
            "lightRowTH"          : ("Light Row Upper Bound", "Light Row Filter Value"),
            "upperDarkTH"         : ("Upper Dark Row Upper Bound", "Upper Dark Row Filter Value"),
            "lowerDarkTH"         : ("Lower Dark Row Upper Bound", "Lower Dark Row Filter Value"),
            "upperUserTH"         : ("Upper User Row Upper Bound", "Upper User Row Filter Value"),
            "lowerUserTH"         : ("Lower User Row Upper Bound", "Lower User Row Filter Value")
        }
        for attr, (eps_key, scale_key) in sampling_advanced_settings.items():
            preset     = getattr(self, attr)
            preset.repCharTol = _float_or_default(settings, eps_key,   preset.repCharTol)
            preset.filterScale = _float_or_default(settings, scale_key, preset.filterScale)

        for attr_name, json_key in preprocessing_advanced_settings.items():
            setattr(self, attr_name, settings.get(json_key, getattr(self, attr_name)))

        for attr_name, (bound_key, delta_key) in preprocessing_background_threshold_settings.items():
            preset = getattr(self, attr_name)
            preset.bound = settings.get(bound_key, preset.bound)
            raw_delta = settings.get(delta_key, preset.delta)
            preset.delta = raw_delta if abs(raw_delta) > 1 else raw_delta


    def __init__(self, settings_from_file: dict = {}):
        # Processing constants
        self.BLACK_TH             = 0.01
        self.WHITE_TH             = 0.99
        self.STAR_MARGIN          = 5
        self.PX_MARGIN            = 10

        # Image Measurement 
        self.col_src_avg_TH       = sampleImagePresets(0.2, 0.99)
        self.row_src_avg_TH       = sampleImagePresets(0.2, 0.99)

        self.col_menu_max_avg_TH  = sampleImagePresets(0.001, 0.99)
        self.row_menu_min_TH      = sampleImagePresets(0.001, 0.97)
        self.col_al_local_min_TH  = sampleImagePresets(0.01, 0.95)
        self.col_al_global_min_TH = sampleImagePresets(0.001, 0.99)
        self.col_al_sep_TH        = sampleImagePresets(0.0005, 0.99)

        # Image Processing
        self.text_menu_TH         = sampleImagePresets(0.01, 0.99)
        self.preproc_attack_avgL  = sampleImagePresets(0.01, 1.00)
        self.new_line_TH          = sampleImagePresets(0.01, 0.97)

        # OCR
        self.no_star_TH           = sampleImagePresets(0.01, 1.00)

        # Preprocessing Presets
        self.lightnessUpperBound = 150
        self.lightnessLowerBound = 0
        self.OUTLINE_UPPER_BGR    = np.array([self.lightnessUpperBound, 
                                              self.lightnessUpperBound,
                                              self.lightnessUpperBound])
        
        self.OUTLINE_LOWER_BGR    = np.array([self.lightnessLowerBound, 
                                              self.lightnessLowerBound,
                                              self.lightnessLowerBound])

        self.BLOB_TH = 0.06

        # Coordinates for sampling at (x0, y0, x1, y1)
        self.lineBgSampling       = [50, 20, 60, 30]
        self.cornerBgSampling     = [0, 0, 5, 5]

        # Background lightness thresholds for different row types
        self.lightRowTH  = backgroundThresholds(0.80, -0.01)
        self.upperDarkTH = backgroundThresholds(0.77, 0.03)
        self.lowerDarkTH = backgroundThresholds(0.70, 0.05)
        self.upperUserTH = backgroundThresholds(0.62, 0.09)
        self.lowerUserTH = backgroundThresholds(0.0,  0.11)

        self.thresholdMap = [
            self.lowerUserTH,  # Checks for >= 0.0
            self.upperUserTH,  # Checks for >= 0.62
            self.lowerDarkTH,  # Checks for >= 0.70
            self.upperDarkTH,  # Checks for >= 0.77
            self.lightRowTH    # Checks for >= 0.80
        ]


        self.TO_DIGIT = str.maketrans({'l':'1', 'I':'1',
                                       '|':'1', 'L':'1',
                                       'T':'1', 'g':'9',
                                       'O':'0', 'o':'0',
                                       'S':'5', 's':'5',
                                       'B':'8', 'W':'11',
                                       'Z':'2', 'z':'2',
                                       'e':'2', 'a':'4',
                                       'd':'1'})
        
        self.DIGIT_GLYPHS = "0-9lLiIoOsSzdeZWagTB|L"

        self.update_from_dict(settings_from_file)