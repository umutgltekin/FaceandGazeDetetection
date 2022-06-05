import cv2
import numpy as np
from utils import Utils


class BlinkingDetection:

    @staticmethod
    def gaze_detection(mask, gray, eye_region):
        cv2.polylines(mask, [eye_region], True, 255, 2)
        cv2.fillPoly(mask, [eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)
        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])
        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        result_eye = cv2.resize(gray_eye, None, fx=5, fy=5)
        mid_point = [(max_y - min_y) / 2, (max_x - min_x) / 2]
        context = {
            'eye': eye,
            'threshold_eye': threshold_eye,
            'result_eye': result_eye,
            'mid_point': mid_point,
            'frame': [min_x, min_y, max_x, max_y]
        }
        return context

    @staticmethod
    def detect_blink(face, predictor, gray_image, right_mask, left_mask):
        landmarks = predictor(gray_image, face)

        left_eye_ratio = Utils.get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = Utils.get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Gaze detection
        right_eye_region = Utils.get_eye_region(landmarks, [42, 43, 44, 45, 46, 47])
        left_eye_region = Utils.get_eye_region(landmarks, [36, 37, 38, 39, 40, 41])

        right = BlinkingDetection.gaze_detection(right_mask, gray_image, right_eye_region)

        left = BlinkingDetection.gaze_detection(left_mask, gray_image, left_eye_region)

        is_blinking = False
        if blinking_ratio > 5.0:
            is_blinking = True

        context = {
            'left': {
                'eye': left['eye'],
                'threshold_eye': left['threshold_eye'],
                'result_eye': left['result_eye'],
                'mid_point': left['mid_point'],
                'frame': left['frame']
            },
            'right': {
                'eye': right['eye'],
                'threshold_eye': right['threshold_eye'],
                'result_eye': right['result_eye'],
                'mid_point': right['mid_point'],
                'frame': right['frame']
            },
            'is_blinking': is_blinking
        }

        return context
