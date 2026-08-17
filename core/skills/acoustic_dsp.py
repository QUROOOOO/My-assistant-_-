from typing import Dict, Any

class AcousticDSP:
    """Mathematical and signal processing formulas for acoustic near-field harmonic tuning."""

    BANDPASS_FREQ = 2190.0  # Center frequency (Hz) for human vocal range (180Hz – 4200Hz)
    BANDPASS_Q = 0.65       # Broad Q factor
    NOISE_FLOOR_DECAY = 0.993
    NOISE_FLOOR_ATTACK = 0.007
    SNR_GATE_FACTOR = 1.4
    BASELINE_GAIN_MULTIPLIER = 12.5  # Amplified baseline sensitivity multiplier

    @staticmethod
    def get_filter_params() -> Dict[str, float]:
        """Returns Web Audio BiquadFilter parameters."""
        return {
            "type": "bandpass",
            "frequency": AcousticDSP.BANDPASS_FREQ,
            "Q": AcousticDSP.BANDPASS_Q
        }

    @staticmethod
    def update_noise_floor(current_floor: float, raw_energy: float) -> float:
        """Adapts ambient noise floor exponentially."""
        return (current_floor * AcousticDSP.NOISE_FLOOR_DECAY +
                raw_energy * AcousticDSP.NOISE_FLOOR_ATTACK)

    @staticmethod
    def calculate_gated_energy(raw_energy: float, noise_floor: float, user_sensitivity: float) -> float:
        """Computes SNR-gated, sensitivity-scaled energy in range [0.0, 1.0]."""
        gate_threshold = noise_floor * AcousticDSP.SNR_GATE_FACTOR
        if raw_energy > gate_threshold:
            effective = (raw_energy - gate_threshold) * user_sensitivity * AcousticDSP.BASELINE_GAIN_MULTIPLIER
            return min(max(effective, 0.0), 1.0)
        return 0.0
