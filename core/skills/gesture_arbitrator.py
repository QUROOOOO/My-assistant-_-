from typing import Dict, Any, List

class GestureArbitrator:
    """Arbitrates mutual exclusivity between gesture states with geometric scale-invariant heuristics."""

    STATE_IDLE = "IDLE"
    STATE_HOVER = "HOVER"
    STATE_GRAB = "GRAB"
    STATE_DUAL_PINCH = "DUAL_PINCH"
    STATE_COMPRESS = "COMPRESS"
    STATE_BLOOM = "BLOOM"
    STATE_SWIPE = "SWIPE"

    # Scale-invariant geometric thresholds (normalized by L_ref)
    PINCH_ENTER_RATIO = 0.30
    PINCH_EXIT_RATIO = 0.42

    FIST_TIP_THRESHOLD = 1.15      # >= 3 fingertips < 1.15 * L_ref
    OPEN_TIP_THRESHOLD = 1.55      # all 4 fingertips > 1.55 * L_ref
    FIST_MEMORY_FRAMES = 15        # 15-frame rolling memory (~250ms)

    VELOCITY_DEADBAND = 0.03       # ||V|| < 0.03 -> V = 0
    SWIPE_VELOCITY_THRESHOLD = 1.6 # ||V|| > 1.6 -> SWIPE
    HOVER_MAX_VELOCITY = 1.4       # ||V|| < 1.4 -> HOVER
