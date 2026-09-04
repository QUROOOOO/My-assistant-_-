from typing import Dict, Any, List

class GestureArbitrator:
    """Arbitrates mutual exclusivity between gesture states with continuous fist/bloom detection."""

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

    # Fist Recognition: all 4 fingertips < 1.1 * L_ref from wrist
    FIST_FINGERTIP_THRESHOLD = 1.10
    FIST_TIGHT_THRESHOLD = 0.50     # E_avg <= 0.50 (tight fist for bloom gating)
    FIST_OPEN_THRESHOLD = 1.20      # E_avg >= 1.20 (open hand)

    # Bloom Trigger: fist held > 200ms, then all fingers > 1.6 * L_ref
    BLOOM_FIST_HOLD_MS = 0.200      # 200ms minimum fist hold
    BLOOM_OPEN_THRESHOLD = 1.60     # All fingers must extend beyond 1.6 * L_ref
    BLOOM_OPEN_WINDOW_MS = 0.150    # Must open within 150ms

    VELOCITY_DEADBAND = 0.03        # ||V|| < 0.03 -> V = 0
    SWIPE_VELOCITY_THRESHOLD = 1.6  # ||V|| > 1.6 -> SWIPE
    HOVER_MAX_VELOCITY = 1.4        # ||V|| < 1.4 -> HOVER
