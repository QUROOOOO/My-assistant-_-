import math
import time
from typing import List, Tuple, Dict, Any, Optional

class OneEuroFilter:
    """Adaptive frequency-domain filter for jitter-free, zero-lag tracking."""
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _smoothing_factor(te: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * te
        return r / (r + 1.0)

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        x = float(x)
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        te = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(te, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class KinematicsValidator:
    """Mathematical helper for Fibonacci distributions, knuckle kinematics, and dual metrics."""

    @staticmethod
    def generate_fibonacci_sphere(point_count: int = 1400, radius: float = 230.0) -> List[Dict[str, float]]:
        """Generates uniformly distributed points on a sphere using the Golden Angle lattice."""
        nodes = []
        golden_angle = math.pi * (3.0 - math.sqrt(5.0)) # ~2.399963 rad
        for i in range(point_count):
            y = 1.0 - (i / (point_count - 1.0)) * 2.0
            phi = math.acos(max(-1.0, min(1.0, y)))
            theta = golden_angle * i
            nx = math.sin(phi) * math.cos(theta)
            ny = math.cos(phi)
            nz = math.sin(phi) * math.sin(theta)
            nodes.append({
                "index": i,
                "theta": theta,
                "phi": phi,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "baseR": radius
            })
        return nodes

    @staticmethod
    def dist_3d(p1, p2) -> float:
        """Computes Euclidean distance between two 3D points."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    @staticmethod
    def calculate_extension_ratio(wrist, mcp, tips: List[Any]) -> float:
        """
        Calculates Normalized Knuckle-to-Fingertip Extension Ratio (E_hand):
        E_hand = (sum_{i in tips} ||P_i - P_wrist||) / (4 * ||P_mcp - P_wrist||)
        """
        knuckle_ref = KinematicsValidator.dist_3d(mcp, wrist)
        if knuckle_ref < 1e-6:
            return 1.0
        tips_dist_sum = sum(KinematicsValidator.dist_3d(tip, wrist) for tip in tips)
        return tips_dist_sum / (len(tips) * knuckle_ref)

    @staticmethod
    def calculate_pinch_ratio(thumb_tip, index_tip, knuckle_ref: float) -> float:
        """Calculates scale-invariant pinch distance ratio."""
        if knuckle_ref < 1e-6:
            return 1.0
        tip_dist = KinematicsValidator.dist_3d(thumb_tip, index_tip)
        return tip_dist / knuckle_ref

    @staticmethod
    def calculate_palm_euler(wrist, middle_mcp, index_mcp, pinky_mcp) -> Tuple[float, float, float]:
        """Calculates 3D Euler angles (pitch, yaw, roll) from palm surface normal."""
        def _vec(p):
            return (float(p.x), float(p.y), float(p.z))
        def _sub(a, b):
            return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
        def _cross(a, b):
            return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])
        def _norm(v):
            m = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            return (v[0]/m, v[1]/m, v[2]/m) if m > 1e-9 else (0.0, 0.0, 0.0)

        v1 = _sub(_vec(middle_mcp), _vec(wrist))
        v2 = _sub(_vec(pinky_mcp), _vec(index_mcp))
        normal = _norm(_cross(v1, v2))

        nx, ny, nz = normal
        pitch = math.asin(max(-1.0, min(1.0, -ny)))
        yaw   = math.atan2(nx, nz)
        roll  = math.atan2(v2[1], math.sqrt(v2[0]**2 + v2[2]**2))
        return pitch, yaw, roll

    @staticmethod
    def calculate_dual_hand_metrics(left_hand: Dict[str, Any], right_hand: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates spatial span, midpoint center, and angle between two hands."""
        dx = right_hand["palm_x"] - left_hand["palm_x"]
        dy = right_hand["palm_y"] - left_hand["palm_y"]
        dist = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        center_x = (left_hand["palm_x"] + right_hand["palm_x"]) / 2.0
        center_y = (left_hand["palm_y"] + right_hand["palm_y"]) / 2.0

        both_pinching = left_hand.get("is_pinching", False) and right_hand.get("is_pinching", False)
        return {
            "dist": dist,
            "angle": angle,
            "center": {"x": center_x, "y": center_y},
            "both_pinching": both_pinching
        }
