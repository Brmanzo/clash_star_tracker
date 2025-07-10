# # File: star_tracker/presets.py
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:                           # only during “mypy / pylance”
    from .state import currentState 

class dataColumn:
    '''Records the absolute position of the column in the original image
    
    Given the relative end point, constructs an object reporting the beginning as
    the previous column's end as well as the resulting width of the column'''
    abs_pos = 0

    def __init__(self, end, begin=0):
        self.begin = dataColumn.abs_pos + begin
        self.end   = end + self.begin
        self.width = self.end - self.begin

        dataColumn.abs_pos += self.width + begin

class sampleImagePresets:
    '''Container for image sampling tuple to use for presets.'''
    def __init__(self, repCharTol: float, filterScale: float):
        self.repCharTol = repCharTol
        self.filterScale = filterScale

class backgroundThresholds:
    '''Container for background lightness thresholds in presets.'''
    def __init__(self, bound: float, delta: float):
        self.bound = bound
        self.delta = delta*255 # Convert delta to pixel value

class imageSlice:
    '''Given a cut of an image and the source width, presents a tuple containing 
    the cut and the overall percentage of the cut in relation to the source width.'''
    def __init__(self, slice: dataColumn | int, end: int, side: str="begin"):
        if isinstance(slice, int):
            self.cut = slice
            if side == "end":
                self.percentage = abs(end - slice) / end
            else:
                self.percentage = abs(self.cut) / end
        elif isinstance(slice, dataColumn):
            self.cut = slice.end
            self.percentage = abs(slice.width) / end

class imageMeasurements:
    """Holds successful measurements for manual cropping if measurement fails."""

    imageMeasurementTable = {
        # cut key                   percentage key
        "menuTopMargin"          : ("menuTopMargin Cut", "menuTopMargin %"),
        "menuBottomMargin"       : ("menuBottomMargin Cut", "menuBottomMargin %"),
        "menuLeftMargin"         : ("menuLeftMargin Cut", "menuLeftMargin %"),
        "menuRightMargin"        : ("menuRightMargin Cut", "menuRightMargin %"),

        "headerEnd"              : ("headerEnd Cut", "headerEnd %"),
        "lineBegin"              : ("lineBegin Cut", "lineBegin %"),
        "lineEnd"                : ("lineEnd Cut", "lineEnd %"),

        "enemyStart"             : ("enemyStart Cut", "enemyStart %"),
        "starsColEnd"            : ("starsColEnd Cut", "starsColEnd %"),
        "percentageBegin"        : ("percentageBegin Cut", "percentageBegin %"),

        "rankCol"                : ("rankCol Cut", "rankCol %"),
        "levelCol"               : ("levelCol Cut", "levelCol %"),
        "playerCol"              : ("playerCol Cut", "playerCol %"),
        "enemyCol"               : ("enemyCol Cut", "enemyCol %"),
        "percentageCol"          : ("percentageCol Cut", "percentageCol %"),
        "starsCol"               : ("starsCol Cut", "starsCol %")
    }

    def to_dict(self) -> dict:
        '''Convert the image measurements to a dictionary for JSON serialization.'''
        out: dict[str, float] = {}
        for attr, (slice_key, percent_key) in self.imageMeasurementTable.items():
            slc = getattr(self, attr)
            if slc is None:
                continue
            out[slice_key]  = round(slc.cut, 5)
            out[percent_key] = round(slc.percentage, 5)
        return out

    def update_from_dict(self, measurements: dict) -> None:
        '''Update the image measurements from a dictionary of measurements.'''
        for attr, (cut_key, pct_key) in self.imageMeasurementTable.items():
            cut_val = measurements.get(cut_key)
            pct_val = measurements.get(pct_key)

            # nothing stored in JSON – skip
            if cut_val is None or pct_val is None:
                continue

            slc = getattr(self, attr)
            if slc is None:                           # create it on-the-fly
                slc = imageSlice(int(cut_val), 1)     # dummy end - we’ll
                setattr(self, attr, slc)              # keep percentage from JSON

            slc.cut        = int(cut_val)
            slc.percentage = float(pct_val)

    def outside_range(self, s: "currentState", measuredPct: float, expectedField: str) -> bool:
        """Check if the actual cut is within the expected range."""
        highRange = s.presets.errMarg
        lowRange = 2 - s.presets.errMarg

        expectedPct = getattr(self, expectedField).percentage
        print(f"Checking if {measuredPct} is outside range of {expectedPct} with low {expectedPct * lowRange} and high {expectedPct * highRange}")
        return not (expectedPct * lowRange <= measuredPct <= expectedPct * highRange)


    def __init__(self, s: "currentState", measurements_from_file: dict = {}):
        self.menuTopMargin: imageSlice | None    = imageSlice(s.menuTopMargin, s.srcDimensions[0]) if (s.menuTopMargin and s.srcDimensions is not None) else None
        self.menuBottomMargin: imageSlice | None = imageSlice(s.menuBottomMargin, s.srcDimensions[0], "end") if (s.menuBottomMargin and s.srcDimensions is not None) else None
        self.menuLeftMargin: imageSlice | None   = imageSlice(s.menuLeftMargin, s.srcDimensions[1]) if (s.menuLeftMargin and s.srcDimensions is not None) else None
        self.menuRightMargin: imageSlice | None  = imageSlice(s.menuRightMargin, s.srcDimensions[1], "end") if (s.menuRightMargin and s.srcDimensions is not None) else None

        self.headerEnd: imageSlice | None        = imageSlice(s.headerEnd, s.menuDimensions[0]) if (s.headerEnd and s.menuDimensions is not None) else None
        self.lineBegin: imageSlice | None        = imageSlice(s.lineBegin, s.menuDimensions[1]) if (s.lineBegin and s.menuDimensions is not None) else None
        self.lineEnd: imageSlice | None          = imageSlice(s.lineEnd, s.menuDimensions[1], "end") if (s.lineEnd and s.menuDimensions is not None) else None

        self.enemyStart: imageSlice | None       = imageSlice(s.enemyStart, s.attackLinesDimensions[1]) if (s.enemyStart and s.attackLinesDimensions is not None) else None
        self.starsColEnd: imageSlice | None      = imageSlice(s.starsColEnd, s.attackLinesDimensions[1], "end") if (s.starsColEnd and s.attackLinesDimensions is not None) else None
        self.percentageBegin: imageSlice | None  = imageSlice(s.percentageBegin, s.attackLinesDimensions[1]) if (s.percentageBegin and s.attackLinesDimensions is not None) else None

        self.rankCol: imageSlice | None          = imageSlice(s.rankCol, s.attackLinesDimensions[1]) if (s.rankCol and s.attackLinesDimensions is not None) else None
        self.levelCol: imageSlice | None         = imageSlice(s.levelCol, s.attackLinesDimensions[1]) if (s.levelCol and s.attackLinesDimensions is not None) else None
        self.playerCol: imageSlice | None        = imageSlice(s.playerCol, s.attackLinesDimensions[1]) if (s.playerCol and s.attackLinesDimensions is not None) else None
        self.enemyCol: imageSlice | None         = imageSlice(s.enemyCol, s.attackLinesDimensions[1]) if (s.enemyCol and s.attackLinesDimensions is not None) else None
        self.percentageCol: imageSlice | None    = imageSlice(s.percentageCol, s.attackLinesDimensions[1]) if (s.percentageCol and s.attackLinesDimensions is not None) else None
        self.starsCol: imageSlice | None         = imageSlice(s.starsCol, s.attackLinesDimensions[1]) if (s.starsCol and s.attackLinesDimensions is not None) else None

        if measurements_from_file:
            self.update_from_dict(measurements_from_file)

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
            "errMarg"             : "Fall-back Tolerance Margin",
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
        self.errMarg = 1.2
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
                                       'd':'1', 'i':'1'})
        
        self.DIGIT_GLYPHS = "0-9lLiIoOsSzdeZWagTB|L"

        self.update_from_dict(settings_from_file)

class gameRulePresets:
    """Holds the game rule presets for Star Bonuses and Penalties."""

    def update_from_dict(self, settings: dict):
        """
        Updates the class attributes from a loaded settings dictionary.
        Uses .get(key, default_value) to safely access the dictionary.
        """
        print("Updating presets from loaded settings...")

        
        gamerule_settings = {
            #  attr                      json-key-key
            "noThreeStarDroppingPenalty"      : "Incomplete Clean Dropping Penalty",
            "noThreeStarDroppingThreshold"    : "Incomplete Clean Dropping Rank Difference",
            "droppingForFirstAttackPenalty"   : "Stealing Lower Target Penalty",
            "droppingForFirstAttackThreshold" : "Stealing Lower Target Rank Difference",
            "successfulJumpBonus"             : "New Star on Higher Target Bonus",
            "successfulJumpThreshold"         : "New Star on Higher Target Rank Difference"
        }

        for attr_name, json_key in gamerule_settings.items():
            setattr(self, attr_name, settings.get(json_key, getattr(self, attr_name)))

    
    def __init__(self, settings_from_file: dict = {}):
        self.noThreeStarDroppingPenalty = -1
        self.noThreeStarDroppingThreshold = -5

        self.droppingForFirstAttackPenalty = "Negate earned stars"
        self.droppingForFirstAttackThreshold = -10

        self.successfulJumpBonus = 1
        self.successfulJumpThreshold = 5

        self.update_from_dict(settings_from_file)
