from typing import Dict, Any, List

class GestureArbitrator:
    """Arbitrates mutual exclusivity between gesture states with state-latched hysteresis heuristics."""

    STATE_IDLE = "IDLE"
    STATE_HOVER = "HOVER"
    STATE_GRAB = "GRAB"
    STATE_DUAL_PINCH = "DUAL_PINCH"
    STATE_COMPRESS = "COMPRESS"
    STATE_BLOOM = "BLOOM"
    STATE_SWIPE = "SWIPE"
    STATE_SNAP = "SNAP"

    # Scale-invariant geometric thresholds (normalized by L_ref)
    PINCH_ENTER_RATIO = 0.30
    PINCH_EXIT_RATIO = 0.42

    # Fist Hysteresis Latch
    FIST_ENTER_THRESHOLD = 1.10    # >= 3 fingertips < 1.10 * L_ref
    FIST_EXIT_THRESHOLD = 1.45     # all 4 fingertips > 1.45 * L_ref
    BLOOM_SNAP_VELOCITY = 2.2      # dE/dt > 2.2 s^-1 within 150ms -> BLOOM

    # Biomechanical Snap Tracking
    SNAP_PRIMED_MAX_DIST = 0.28    # Thumb & Middle tip distance < 0.28 * L_ref
    SNAP_DOWNWARD_DELTA_Y = 0.08   # Delta Y slip > 0.08 within 160ms
    SNAP_VISION_CONF_THRESHOLD = 0.62
    SNAP_COOLDOWN_SEC = 2.0        # Refractory period

    VELOCITY_DEADBAND = 0.03       # ||V|| < 0.03 -> V = 0
    SWIPE_VELOCITY_THRESHOLD = 1.6 # ||V|| > 1.6 -> SWIPE
    HOVER_MAX_VELOCITY = 1.4       # ||V|| < 1.4 -> HOVER
