import cv2

# Enable verbose debugging, fine-grained debugging booleans below are likely to
# be more useful.
DEBUG_VERBOSE = False

# -----------------------------
# Camera capture settings
# -----------------------------

CAPTURE_WIDTH = 960
CAPTURE_HEIGHT = 720
CAPTURE_VFLIP = True

MAX_FPS = 4

# -----------------------------
# Coroner detection
# -----------------------------

DEBUG_CORNERS = False

# Board has red circle stickers on the outline
BOARD_MODE_RED_CIRCLES = False

# Board has a red boarder (newer scrabble board)
BOARD_MODE_RED_BORDER = False

# Board has ArUco fiducial markers on corners. This is going to be the most
# reliable detection method. Print out corner_markers.pdf and apply these
# labels to the corners.
BOARD_MODE_MARKERS = True

CORNER_ERODE_RAD = 3
CORNER_DILATE_RAD = 3
CORNER_BLUR_RAD = 5

CORNER_THRESH_PARAM = 41
CORNER_BLOCK_SIZE = 9

# Reject corners if the board's edge lengths deviate by more than this amount.
CORNER_SIDE_DEV_THRESH = 0.10

# Hysteresis configuration
CORNER_HISTORY_COUNT = 10
CORNER_MOVE_REJECT_THRESH = 100

# Position of game board relative to the corners. Adjustments are in 1/1000
# units relative to the corners.

# Adjustments for Jeff's board [ArUco markers]
# TL_X = 10
# TL_Y = 5
# TR_X = 10
# TR_Y = 5
# BL_X = 10
# BL_Y = 5
# BR_X = 10
# BR_Y = 5

# Adjustments for Parents' board [ArUco markers]
TL_X = 14
TL_Y = 5
TR_X = 10
TR_Y = 5
BL_X = 14
BL_Y = 10
BR_X = 10
BR_Y = 10

## Adjustments for parents' board [red circles]
# TL_X = 54
# TL_Y = 2
# TR_X = 20
# TR_Y = 25
# BL_X = 48
# BL_Y = 18
# BR_X = 20
# BR_Y = 22

## Adjustments for Jeff's board [red border]
#  TL_X = 190
#  TL_Y = 40
#  TR_X = 40
#  TR_Y = 40
#  BL_X = 195
#  BL_Y = 40
#  BR_X = 40
#  BR_Y = 40

# -----------------------------
# Letter detection
# -----------------------------

DEBUG_LETTERS = False

TRAIN = False
TRAIN_APPEND = True

# Per-letter processing size (pixels)
LETTER_SIZE = 48

# Per-letter training size (pixels)
LETTER_TRAIN_SIZE = 18

# Fraction of the letter's area use for training (cutting off the edges)
LETTER_TRAIN_SUBPIX_FRAC = 0.75

LETTER_COLORSPACE = cv2.COLOR_RGB2HSV
LETTER_CHANNEL = 2  # V of HSV

# Adjustments for parents' board
LETTER_BLOCK = 45
LETTER_THRESH = 19
LETTER_BLUR = 5

# Adjustments for Jeff's board
#LETTER_BLOCK = 47
#LETTER_THRESH = 31
#LETTER_BLUR = 5

# Expand the board by this amount to account for letters off the edge.
LETTER_PAD_FRAC = 0.5

# Allow letters to be out of position (misaligned) by this fraction.
LETTER_MAX_SHIFT_FRAC = 0.25

# Reasonableness filters for per-letter contours
LETTER_CONTOUR_MIN_FRAC = 0.030
LETTER_CONTOUR_MAX_FRAC = 0.650
LETTER_TEXT_RATIO = 1.4
LETTER_MAX_FILL=0.8

# Amount of averaging to do over time for letter detections.
BOARD_LETTER_HISTORY_SIZE = 10

# -----------------------------
# Blank detection
# -----------------------------

DEBUG_BLANKS = False

# NOTE: H channel of HSV also works well here.
BLANK_COLORSPACE = cv2.COLOR_RGB2GRAY
BLANK_CHANNEL = None

# Fraction of the letter to use for blank detection.
BLANK_PATCH_FRAC = 0.40


# Require higher confidence for blank detection in the averaged board
BLANK_REQ_PERCENT = 0.65

BLANK_NEIGHBORS = 10

# For parents' board
# Blank tiles will be smooth. Reject tiles above this threshold.
BLANK_COEF_VAR_MAX = 8.0

BLANK_Z_THRESH = 8

# For Jeff's board
## Blank tiles will be smooth. Reject tiles above this threshold.
#BLANK_COEF_VAR_MAX = 2.9
#
#BLANK_Z_THRESH = 3.0

# -----------------------------
# Game settings
# -----------------------------

# Number of letters that make up the scrabble board.
# NOTE: not fully supported to change this. Scoring only works for 15x15. Maybe
# super scrabble some day.
BOARD_SIZE = 15

WARN_TIME = 3
ALARM_TIME = 4
