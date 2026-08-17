from typing import Dict, Any, List

class GestureArbitrator:
    """Arbitrates mutual exclusivity between gesture states with strict 2-phase bloom gating."""

    STATE_IDLE = "IDLE"
    STATE_HOVER = "HOVER"
    STATE_DUAL_PINCH_ZOOM = "DUAL_PINCH_ZOOM"
    STATE_COMPRESS = "COMPRESS"
    STATE_BLOOM = "BLOOM"
    STATE_SLAP = "SLAP"

    # Strict 2-phase Extension ratio thresholds
    FIST_CHARGE_THRESHOLD = 0.72       # Must be < 0.72 + thumb tucked
    FIST_CHARGE_MIN_FRAMES = 5         # At least 5 frames (~80ms)
    FIST_EXIT_THRESHOLD = 1.05
    BLOOM_VELOCITY_THRESHOLD = 3.8     # dE/dt in s^-1 (explosive snap trigger)
    SLOW_OPEN_VELOCITY_LIMIT = 2.0     # dE/dt in s^-1 (safe slow open reversion)

    PINCH_ENTER_RATIO = 0.20
    PINCH_EXIT_RATIO = 0.30

    @staticmethod
    def arbitrate_state(
        hands: List[Dict[str, Any]],
        dual_pinch: bool,
        global_bloom: bool,
        global_compress: bool,
        slap_active: bool
    ) -> str:
        """Determines the single authoritative gesture state."""
        if global_bloom:
            return GestureArbitrator.STATE_BLOOM

        if dual_pinch:
            return GestureArbitrator.STATE_DUAL_PINCH_ZOOM

        if global_compress:
            return GestureArbitrator.STATE_COMPRESS

        if slap_active:
            return GestureArbitrator.STATE_SLAP

        if len(hands) > 0:
            return GestureArbitrator.STATE_HOVER

        return GestureArbitrator.STATE_IDLE
