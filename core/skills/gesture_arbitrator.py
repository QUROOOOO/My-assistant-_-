from typing import Dict, Any, List

class GestureArbitrator:
    """Arbitrates mutual exclusivity between gesture states with Post-Pose Delta snap detection."""

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

    # Fist Recognition: all 4 fingertips < 1.1 * L_ref from wrist
    FIST_FINGERTIP_THRESHOLD = 1.10
    FIST_TIGHT_THRESHOLD = 0.50     # E_avg <= 0.50 (tight fist for bloom gating)
    FIST_OPEN_THRESHOLD = 1.20      # E_avg >= 1.20 (open hand)

    # Bloom Trigger: fist held > 200ms, then all fingers > 1.6 * L_ref within 150ms
    BLOOM_FIST_HOLD_MS = 0.200      # 200ms minimum fist hold
    BLOOM_OPEN_THRESHOLD = 1.60     # All fingers must extend beyond 1.6 * L_ref
    BLOOM_OPEN_WINDOW_MS = 0.150    # Must open within 150ms

    # Post-Pose Delta State Matching Snap Detection
    SNAP_PREPOSE_THUMB_MIDDLE = 0.25   # Pre-Pose: ||P4 - P12|| / L_ref < 0.25
    SNAP_POSTPOSE_THUMB_INDEX = 0.30   # Post-Pose: ||P4 - P5|| / L_ref < 0.30
    SNAP_POSTPOSE_MIDDLE_PALM = 0.80   # Post-Pose: ||P12 - P0|| / L_ref < 0.80
    SNAP_TRANSITION_WINDOW_MS = 0.180  # Pre-Pose -> Post-Pose within 180ms
    SNAP_COOLDOWN_SEC = 2.0            # Refractory period

    VELOCITY_DEADBAND = 0.03        # ||V|| < 0.03 -> V = 0
    SWIPE_VELOCITY_THRESHOLD = 1.6  # ||V|| > 1.6 -> SWIPE
    HOVER_MAX_VELOCITY = 1.4        # ||V|| < 1.4 -> HOVER
