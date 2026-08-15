import cv2
import numpy as np
import torch

class ZebraDetector:
    _model = None
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _roi_cache = None
    _roi_frame_counter = 0
    _roi_recompute_every = 60
    _polygon_cache = None
    _polygon_frame_counter = 0
    _polygon_recompute_every = 15

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            try:
                from ultralytics import YOLO
                cls._model = YOLO("yolo11x-seg.pt").to(cls._device)
            except Exception:
                pass
        return cls._model

    @classmethod
    def _get_road_roi(cls, frame: np.ndarray):
        h, w = frame.shape[:2]
        seg_model = cls._get_model()
        if seg_model is None:
            return int(h * 0.55), int(h * 0.95), int(w * 0.15), int(w * 0.85)

        if cls._roi_cache is not None and cls._roi_frame_counter < cls._roi_recompute_every:
            cls._roi_frame_counter += 1
            return cls._roi_cache

        results = seg_model(frame, verbose=False, classes=[0, 2, 3, 5, 7], device=cls._device)
        bottom_obstacle = h * 0.55
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                for box in boxes:
                    _, y2, _, _ = box
                    if y2 < h * 0.9:
                        bottom_obstacle = max(bottom_obstacle, float(y2))
        roi_ymin = max(int(bottom_obstacle), int(h * 0.50))
        roi_ymin = min(roi_ymin, int(h * 0.80))
        roi_ymax = int(h * 0.95)
        roi_xmin, roi_xmax = int(w * 0.15), int(w * 0.85)
        cls._roi_cache = (roi_ymin, roi_ymax, roi_xmin, roi_xmax)
        cls._roi_frame_counter = 1
        return cls._roi_cache

    @classmethod
    def reset_cache(cls):
        cls._roi_cache = None
        cls._roi_frame_counter = 0
        cls._polygon_cache = None
        cls._polygon_frame_counter = 0

    @classmethod
    def detect_zebra_crossing(cls, frame: np.ndarray) -> bool:
        polygon = cls.get_zebra_polygon(frame)
        return polygon is not None and len(polygon) > 0

    @classmethod
    def get_zebra_polygon(cls, frame: np.ndarray) -> list:
        if frame is None or not isinstance(frame, np.ndarray):
            return None
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return None

        if cls._polygon_cache is not None and cls._polygon_frame_counter < cls._polygon_recompute_every:
            cls._polygon_frame_counter += 1
            return cls._polygon_cache
        cls._polygon_frame_counter = 1

        roi_ymin, roi_ymax, roi_xmin, roi_xmax = cls._get_road_roi(frame)
        roi = frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=15)
        if lines is None:
            return None

        valid_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                angle = 90.0
            else:
                angle = np.degrees(np.arctan2(y2 - y1, dx))
            angle = np.abs(angle)
            if 5 < angle < 175 and angle != 90.0:
                valid_points.append([x1 + roi_xmin, y1 + roi_ymin])
                valid_points.append([x2 + roi_xmin, y2 + roi_ymin])

        if len(valid_points) < 8:
            return None

        pts = np.array(valid_points, dtype=np.int32)
        hull = cv2.convexHull(pts)
        polygon = hull.reshape(-1, 2).tolist()
        cls._polygon_cache = polygon
        return polygon

    @classmethod
    def is_point_in_zebra(cls, polygon: list, point: tuple) -> bool:
        """
        Checks if a point (x, y) is inside the zebra crossing polygon.
        """
        if polygon is None or len(polygon) == 0:
            return False
        contour = np.array(polygon, dtype=np.int32)
        dist = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
        return dist >= 0


