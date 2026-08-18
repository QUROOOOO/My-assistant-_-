from typing import Dict, Any, List

class GestureArbitrator:
    """Arbitrates mutual exclusivity between gesture states with continuous flexion and biomechanical isolation."""

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

    # Continuous Knuckle Extension Thresholds
    FIST_TIGHT_THRESHOLD = 0.50     # E_avg <= 0.50 (tight fist)
    FIST_OPEN_THRESHOLD = 1.20      # E_avg >= 1.20 (open hand)
    BLOOM_EXP_VELOCITY = 3.8        # dE/dt > 3.8 s^-1 explosive burst

    # Biomechanical Snap Tracking & Isolation
    SNAP_PRIMED_MAX_DIST = 0.28     # Thumb & Middle tip distance < 0.28 * L_ref
    SNAP_FINGER_EXT_THRESHOLD = 1.0 # Ring (16) and Pinky (20) must be > 1.0 * L_ref
    SNAP_DOWNWARD_DELTA_Y = 0.06    # Delta Y slip > 0.06 within 160ms
    SNAP_VISION_CONF_THRESHOLD = 0.62
    SNAP_COOLDOWN_SEC = 2.0         # Refractory period

    VELOCITY_DEADBAND = 0.03        # ||V|| < 0.03 -> V = 0
    SWIPE_VELOCITY_THRESHOLD = 1.6  # ||V|| > 1.6 -> SWIPE
    HOVER_MAX_VELOCITY = 1.4        # ||V|| < 1.4 -> HOVER
