"""Simple color-based target detection from camera images."""

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .utils import setup_logger


class TargetDetector:
    """Detects targets in RGB images using HSV color filtering.

    Identifies objects matching a specified color range (default: red)
    and returns their bounding boxes and centroids.

    Attributes:
        hsv_lower: Lower HSV bound for color detection.
        hsv_upper: Upper HSV bound for color detection.
        min_area: Minimum contour area to count as a detection.
    """

    def __init__(self, config: Dict):
        """Initialize the target detector.

        Args:
            config: Configuration dict with target detection parameters.
        """
        self.hsv_lower = np.array(config.get("target_color_hsv_lower", [0, 120, 70]))
        self.hsv_upper = np.array(config.get("target_color_hsv_upper", [10, 255, 255]))
        self.min_area = config.get("target_detection_min_area", 500)
        self.logger = setup_logger("TargetDetector")

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect targets in an RGB image.

        Converts the image to HSV, applies color thresholding,
        finds contours, and returns detections above the minimum area.

        Args:
            image: RGB image as numpy array (H, W, 3).

        Returns:
            List of detection dicts, each containing:
                - 'centroid': (cx, cy) pixel coordinates
                - 'bbox': (x, y, w, h) bounding rectangle
                - 'area': contour area in pixels
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            detections.append({
                "centroid": (cx, cy),
                "bbox": (x, y, w, h),
                "area": area
            })

        if detections:
            self.logger.info(f"Detected {len(detections)} target(s)")

        return detections

    def annotate_image(self, image: np.ndarray,
                       detections: List[Dict]) -> np.ndarray:
        """Draw detection bounding boxes and centroids on the image.

        Args:
            image: Original RGB image.
            detections: List of detection dicts from detect().

        Returns:
            Annotated image copy.
        """
        annotated = image.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            cx, cy = det["centroid"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(annotated, f"Target ({cx},{cy})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
        return annotated
