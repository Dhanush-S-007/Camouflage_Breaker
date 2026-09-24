# ============================================================
# Camouflage Breaker - Object Presence Gate
# ============================================================

import cv2
import numpy as np


class ObjectPresenceGate:

    def __init__(
        self,
        min_area_ratio=0.008,
        min_component_ratio=0.004,
        min_mean_probability=0.58,
        min_max_probability=0.70
    ):
        self.min_area_ratio = min_area_ratio
        self.min_component_ratio = min_component_ratio
        self.min_mean_probability = min_mean_probability
        self.min_max_probability = min_max_probability

    def evaluate(
        self,
        probability_map,
        binary_mask
    ):
        """
        Decide whether the segmentation represents
        a real object or a false positive.

        Returns:
            detected: bool
            information: dict
        """

        if probability_map is None:
            return False, {
                "reason": "No probability map"
            }

        if binary_mask is None:
            return False, {
                "reason": "No binary mask"
            }

        if probability_map.size == 0:
            return False, {
                "reason": "Empty probability map"
            }

        if binary_mask.size == 0:
            return False, {
                "reason": "Empty binary mask"
            }

        # ----------------------------------------------------
        # Convert mask to uint8
        # ----------------------------------------------------

        mask = (
            binary_mask > 0
        ).astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Total mask area
        # ----------------------------------------------------

        total_pixels = mask.shape[0] * mask.shape[1]

        object_pixels = int(
            np.sum(mask)
        )

        area_ratio = (
            object_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        # ----------------------------------------------------
        # No predicted pixels
        # ----------------------------------------------------

        if object_pixels == 0:

            return False, {
                "reason": "No segmentation region",
                "area_ratio": 0.0,
                "largest_component_ratio": 0.0,
                "mean_probability": 0.0,
                "max_probability": 0.0
            }

        # ----------------------------------------------------
        # Connected components
        # ----------------------------------------------------

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

        largest_component_pixels = 0

        if num_labels > 1:

            component_sizes = stats[
                1:,
                cv2.CC_STAT_AREA
            ]

            largest_component_pixels = int(
                np.max(component_sizes)
            )

        largest_component_ratio = (
            largest_component_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Probability values inside predicted region
        # ----------------------------------------------------

        probability_map = np.asarray(
            probability_map,
            dtype=np.float32
        )

        probability_map = np.clip(
            probability_map,
            0.0,
            1.0
        )

        region_probabilities = probability_map[
            mask > 0
        ]

        if len(region_probabilities) == 0:

            mean_probability = 0.0
            max_probability = 0.0

        else:

            mean_probability = float(
                np.mean(region_probabilities)
            )

            max_probability = float(
                np.max(region_probabilities)
            )

        # ----------------------------------------------------
        # Presence checks
        # ----------------------------------------------------

        area_ok = (
            area_ratio >= self.min_area_ratio
        )

        component_ok = (
            largest_component_ratio
            >= self.min_component_ratio
        )

        mean_probability_ok = (
            mean_probability
            >= self.min_mean_probability
        )

        max_probability_ok = (
            max_probability
            >= self.min_max_probability
        )

        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------
        #
        # Require:
        #
        # 1. reasonable total object area
        # 2. reasonable connected component
        # 3. reliable average segmentation probability
        # 4. strong peak probability
        #
        # This prevents tiny/random false-positive masks
        # from reaching ResNet50.
        # ----------------------------------------------------

        detected = (
            area_ok
            and component_ok
            and mean_probability_ok
            and max_probability_ok
        )

        information = {
            "area_ratio": float(area_ratio),
            "largest_component_ratio": float(
                largest_component_ratio
            ),
            "mean_probability": float(
                mean_probability
            ),
            "max_probability": float(
                max_probability
            ),
            "area_ok": bool(area_ok),
            "component_ok": bool(component_ok),
            "mean_probability_ok": bool(
                mean_probability_ok
            ),
            "max_probability_ok": bool(
                max_probability_ok
            ),
            "detected": bool(detected)
        }

        if detected:

            information["reason"] = (
                "Reliable object segmentation"
            )

        else:

            information["reason"] = (
                "Segmentation rejected as false positive"
            )

        return detected, information