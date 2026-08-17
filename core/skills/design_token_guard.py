from typing import Dict, Any, List

class DesignTokenGuard:
    """Validates CSS design token rules, layout constraints, and animation contracts."""

    REQUIRED_THEMES = ["dark", "light"]

    TOKEN_SPEC = {
        "dark": {
            "--bg": "#000000",
            "--text": "#ffffff",
            "--text-dim": "#71717a",
            "--border": "rgba(255, 255, 255, 0.12)",
            "--hud-bg": "rgba(0, 0, 0, 0.7)",
        },
        "light": {
            "--bg": "#ffffff",
            "--text": "#09090b",
            "--text-dim": "#a1a1aa",
            "--border": "rgba(0, 0, 0, 0.1)",
            "--hud-bg": "rgba(255, 255, 255, 0.8)",
        }
    }

    LAYOUT_RULES = {
        "camera_feed_toggle_hover": "transform: scale(1.06); (non-rotating)",
        "sensor_hud_title": "LIVE FEED centered with letter-spacing: 2px; font-weight: 700;",
        "settings_manual_section": "max-height: calc(100vh - 340px); overflow-y: auto; 4px scrollbar",
        "capsule_slider": "height: 14px; border-radius: 9999px; groove styling"
    }

    @staticmethod
    def validate_theme_tokens(tokens: Dict[str, str], theme: str = "dark") -> List[str]:
        """Checks for missing required design tokens."""
        missing = []
        expected = DesignTokenGuard.TOKEN_SPEC.get(theme, {})
        for key in expected:
            if key not in tokens:
                missing.append(key)
        return missing
