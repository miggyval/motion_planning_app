# constants.py

import math

# ------------------------
# Canvas & grid config
# ------------------------
INIT_ROWS = 40
INIT_COLS = 60
INIT_CELL = 16

# Fixed canvas size derived from the initial grid so the window never changes
CANVAS_H = INIT_ROWS * INIT_CELL
CANVAS_W = INIT_COLS * INIT_CELL

MARGIN = 1              # grid line thickness
ANIM_DELAY = 0.0005     # seconds between explored draws
WINDOW = "Pathfinding Demo"

# Grid size constraints (keeps cells at least ~2 px high/wide on this canvas)
MIN_ROWS, MIN_COLS = 10, 15
MAX_ROWS, MAX_COLS = 80, 120

# Colors (BGR)
COL_BG       = (245, 245, 245)
COL_GRID     = (210, 210, 210)
COL_OBS      = (40, 40, 40)
COL_START    = (0, 180, 0)
COL_GOAL     = (0, 0, 205)
COL_PATH     = (255, 100, 0)
COL_EXPLORED = (190, 220, 255)
COL_TEXT     = (30, 30, 30)
COL_HINT     = (90, 90, 90)
COL_PREVIEW  = (160, 160, 160)
COL_PRM_NODE = (60, 60, 255)
COL_PRM_EDGE = (220, 220, 220)
COL_PRM_VIS  = (100, 170, 50)

# Costmap colors
COL_COST_CYAN  = (255, 255, 0)    # very high cost
COL_COST_PINK  = (203, 192, 255)  # high shoulder
COL_COST_WHITE = (255, 255, 255)  # low shoulder

# Pens
PEN_START, PEN_GOAL, PEN_DRAW, PEN_ERASE, PEN_LINEDRAW, PEN_LINEERASE = 0, 1, 2, 3, 4, 5
PEN_NAMES = ["Start", "Goal", "Draw", "Erase", "Line-Draw", "Line-Erase"]
