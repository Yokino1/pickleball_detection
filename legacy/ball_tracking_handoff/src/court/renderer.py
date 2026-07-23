"""
Court renderer — draws mini-court, keypoints, and projected court lines.

Key design rule:
    The mini-court is ALWAYS drawn, even with zero keypoints.
    Image-space court overlay is only drawn when homography is reliable.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .layout import CourtLayout
from .observation import CourtObservation
from .projector import CourtProjector


# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------

C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_BLUE = (255, 0, 0)
C_GREEN = (0, 255, 0)
C_RED = (0, 0, 255)
C_YELLOW = (0, 255, 255)
C_ORANGE = (0, 165, 255)
C_GREY = (128, 128, 128)
C_CYAN = (255, 255, 0)


class CourtRenderer:
    """Draws mini-court, image keypoints, and projected court overlay."""

    def __init__(
        self,
        mini_court_width: int = 280,
        mini_court_height: int = 420,
        margin_right: int = 20,
        margin_top: int = 20,
    ):
        self.mini_court_width = mini_court_width
        self.mini_court_height = mini_court_height
        self.margin_right = margin_right
        self.margin_top = margin_top

    # ------------------------------------------------------------------
    # Mini-court
    # ------------------------------------------------------------------

    def draw_mini_court(
        self,
        frame: np.ndarray,
        court_layout: CourtLayout,
        ball_court_xy: Optional[tuple[float, float]] = None,
        projection_status: str = "none",
    ) -> np.ndarray:
        """Draw the standard mini-court in the top-right corner.

        Always draws — does not depend on keypoints or homography.

        Args:
            frame: BGR image to draw on.
            court_layout: Standard court geometry.
            ball_court_xy: Ball position in court feet, or None.
            projection_status: Current projection status string.

        Returns:
            The frame (modified in-place).
        """
        fh, fw = frame.shape[:2]

        mc_w = self.mini_court_width
        mc_h = self.mini_court_height
        x1 = fw - mc_w - self.margin_right
        y1 = self.margin_top
        x2 = x1 + mc_w
        y2 = y1 + mc_h

        # White canvas
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_WHITE, cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_BLACK, 1)

        # Court lines
        mini_info = court_layout.mini_court_points(mc_w, mc_h)
        for (lx1, ly1), (lx2, ly2) in mini_info["lines"]:
            cv2.line(
                frame,
                (x1 + lx1, y1 + ly1),
                (x1 + lx2, y1 + ly2),
                C_BLACK, 1, cv2.LINE_AA,
            )

        # Net highlight
        kps = mini_info["keypoints_px"]
        net_left = kps[6]
        net_right = kps[7]
        cv2.line(
            frame,
            (x1 + net_left[0], y1 + net_left[1]),
            (x1 + net_right[0], y1 + net_right[1]),
            C_BLUE, 2, cv2.LINE_AA,
        )

        # Ball dot (if we have court coordinates)
        if ball_court_xy is not None:
            bx, by = court_layout.court_ft_to_mini_px(ball_court_xy, mini_info)
            cv2.circle(
                frame, (x1 + bx, y1 + by), 5, C_YELLOW, -1, cv2.LINE_AA,
            )
            cv2.circle(
                frame, (x1 + bx, y1 + by), 6, C_BLACK, 1, cv2.LINE_AA,
            )

        # Status label
        if projection_status in ("unavailable", "none", "cached"):
            label = (
                "no projection" if projection_status in ("unavailable", "none")
                else "cached H"
            )
            cv2.putText(
                frame, label,
                (x1 + 4, y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GREY, 1, cv2.LINE_AA,
            )

        return frame

    # ------------------------------------------------------------------
    # Image-space keypoints
    # ------------------------------------------------------------------

    def draw_image_keypoints(
        self,
        frame: np.ndarray,
        observation: CourtObservation,
    ) -> np.ndarray:
        """Draw known court keypoints on the main image.

        Green = high confidence (≥0.5), Orange = low confidence.
        """
        for kp_idx, (px, py) in observation.keypoints.items():
            conf = observation.confidences.get(kp_idx, 0.0)
            colour = C_GREEN if conf >= 0.5 else C_ORANGE
            cv2.circle(
                frame, (int(px), int(py)), 4, colour, -1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, str(kp_idx),
                (int(px) + 6, int(py) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA,
            )
        return frame

    # ------------------------------------------------------------------
    # Projected court overlay (image space)
    # ------------------------------------------------------------------

    def draw_image_court_lines(
        self,
        frame: np.ndarray,
        projector: CourtProjector,
        court_layout: CourtLayout,
    ) -> np.ndarray:
        """Draw court lines on the main image using inverse homography.

        Only call this when ``projector.is_available`` is True.
        """
        H = projector.homography
        if H is None:
            return frame

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return frame

        for i, j in court_layout.line_segments:
            ft1 = court_layout.canonical_keypoints[i]
            ft2 = court_layout.canonical_keypoints[j]

            # Project court-ft points back to image pixels
            p1 = self._project_inverse(ft1, H_inv)
            p2 = self._project_inverse(ft2, H_inv)
            if p1 is None or p2 is None:
                continue

            cv2.line(
                frame,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                C_CYAN, 1, cv2.LINE_AA,
            )

        return frame

    @staticmethod
    def _project_inverse(
        court_ft: tuple[float, float],
        H_inv: np.ndarray,
    ) -> Optional[tuple[float, float]]:
        """Project a court-ft point back to image pixels."""
        ft = np.array([[[court_ft[0], court_ft[1]]]], dtype=np.float32)
        try:
            im = cv2.perspectiveTransform(ft, H_inv)
            return (float(im[0][0][0]), float(im[0][0][1]))
        except cv2.error:
            return None
