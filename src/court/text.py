"""Cached Chinese status banners for the court projection panel."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
)


class ChineseStatusBanner:
    """Render each small status banner once, then reuse its BGR pixels."""

    def __init__(
        self,
        *,
        font_path: str | None = None,
        font_size: int = 48,
    ):
        self.font_path = self._resolve_font_path(font_path)
        self.font_size = max(20, int(font_size))
        self._cache: dict[
            tuple[int, str, tuple[int, int, int]],
            np.ndarray,
        ] = {}
        self._inline_cache: dict[
            tuple[str, int, tuple[int, int, int]],
            np.ndarray,
        ] = {}

    @property
    def chinese_available(self) -> bool:
        return self.font_path is not None

    def draw(
        self,
        panel: np.ndarray,
        *,
        text_zh: str,
        fallback_text: str,
        color_bgr: tuple[int, int, int],
    ) -> None:
        panel_width = int(panel.shape[1])
        cache_key = (panel_width, text_zh, color_bgr)
        banner = self._cache.get(cache_key)
        if banner is None:
            banner = self._build_banner(
                panel_width,
                text_zh,
                fallback_text,
                color_bgr,
            )
            self._cache[cache_key] = banner
        banner_height, banner_width = banner.shape[:2]
        x = max(0, (panel_width - banner_width) // 2)
        y = 16
        y_end = min(panel.shape[0], y + banner_height)
        x_end = min(panel_width, x + banner_width)
        if y_end > y and x_end > x:
            panel[y:y_end, x:x_end] = banner[
                : y_end - y,
                : x_end - x,
            ]

    def _build_banner(
        self,
        panel_width: int,
        text_zh: str,
        fallback_text: str,
        color_bgr: tuple[int, int, int],
    ) -> np.ndarray:
        banner_width = max(220, panel_width - 32)
        banner_height = self.font_size + 34
        banner = np.full(
            (banner_height, banner_width, 3),
            (12, 12, 12),
            dtype=np.uint8,
        )
        cv2.rectangle(
            banner,
            (1, 1),
            (banner_width - 2, banner_height - 2),
            color_bgr,
            2,
            cv2.LINE_AA,
        )
        if self.font_path is None:
            self._draw_fallback(banner, fallback_text, color_bgr)
            return banner
        try:
            from PIL import Image, ImageDraw, ImageFont

            rgb = cv2.cvtColor(banner, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(
                str(self.font_path),
                self.font_size,
            )
            label = f"球状态：{text_zh}"
            bounds = draw.textbbox((0, 0), label, font=font)
            text_width = bounds[2] - bounds[0]
            text_height = bounds[3] - bounds[1]
            x = max(8, (banner_width - text_width) // 2)
            y = max(4, (banner_height - text_height) // 2 - bounds[1])
            color_rgb = (
                int(color_bgr[2]),
                int(color_bgr[1]),
                int(color_bgr[0]),
            )
            draw.text((x, y), label, font=font, fill=color_rgb)
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        except (ImportError, OSError, ValueError):
            self._draw_fallback(banner, fallback_text, color_bgr)
            return banner

    def draw_text(
        self,
        panel: np.ndarray,
        *,
        text_zh: str,
        fallback_text: str,
        origin: tuple[int, int],
        color_bgr: tuple[int, int, int],
        font_size: int,
    ) -> None:
        """Draw cached inline CJK text with an ASCII OpenCV fallback."""
        x, y = (int(origin[0]), int(origin[1]))
        size = max(20, int(font_size))
        if self.font_path is None:
            self._draw_inline_fallback(
                panel,
                fallback_text,
                origin=(x, y),
                color_bgr=color_bgr,
                font_size=size,
            )
            return

        cache_key = (str(text_zh), size, color_bgr)
        overlay = self._inline_cache.get(cache_key)
        if overlay is None:
            try:
                from PIL import Image, ImageDraw, ImageFont

                font = ImageFont.truetype(str(self.font_path), size)
                probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
                draw = ImageDraw.Draw(probe)
                bounds = draw.textbbox((0, 0), str(text_zh), font=font)
                width = max(1, bounds[2] - bounds[0] + 6)
                height = max(1, bounds[3] - bounds[1] + 6)
                image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                color_rgba = (
                    int(color_bgr[2]),
                    int(color_bgr[1]),
                    int(color_bgr[0]),
                    255,
                )
                draw.text(
                    (3 - bounds[0], 3 - bounds[1]),
                    str(text_zh),
                    font=font,
                    fill=color_rgba,
                )
                overlay = cv2.cvtColor(
                    np.asarray(image),
                    cv2.COLOR_RGBA2BGRA,
                )
                self._inline_cache[cache_key] = overlay
            except (ImportError, OSError, ValueError):
                self._draw_inline_fallback(
                    panel,
                    fallback_text,
                    origin=(x, y),
                    color_bgr=color_bgr,
                    font_size=size,
                )
                return

        height, width = overlay.shape[:2]
        if x >= panel.shape[1] or y >= panel.shape[0]:
            return
        x_end = min(panel.shape[1], x + width)
        y_end = min(panel.shape[0], y + height)
        if x_end <= x or y_end <= y:
            return
        visible = overlay[: y_end - y, : x_end - x]
        alpha = visible[:, :, 3:4].astype(np.float32) / 255.0
        target = panel[y:y_end, x:x_end].astype(np.float32)
        blended = visible[:, :, :3].astype(np.float32) * alpha + target * (
            1.0 - alpha
        )
        panel[y:y_end, x:x_end] = blended.astype(np.uint8)

    @staticmethod
    def _draw_inline_fallback(
        panel: np.ndarray,
        text: str,
        *,
        origin: tuple[int, int],
        color_bgr: tuple[int, int, int],
        font_size: int,
    ) -> None:
        x, y = origin
        cv2.putText(
            panel,
            text,
            (x, y + font_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.55, font_size / 38.0),
            color_bgr,
            max(2, font_size // 18),
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_fallback(
        banner: np.ndarray,
        fallback_text: str,
        color_bgr: tuple[int, int, int],
    ) -> None:
        label = f"BALL STATUS: {fallback_text.upper()}"
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        x = max(8, (banner.shape[1] - text_width) // 2)
        y = max(text_height + 4, (banner.shape[0] + text_height) // 2)
        cv2.putText(
            banner,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color_bgr,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _resolve_font_path(configured: str | None) -> Path | None:
        if configured:
            path = Path(configured).expanduser()
            return path if path.is_file() else None
        for path in _FONT_CANDIDATES:
            if path.is_file():
                return path
        return None
