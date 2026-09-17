import cv2
import numpy as np

class TrashBagCSRTTracker:
    """
    Fallback Single Object Tracker using OpenCV CSRT.
    When YOLO fails to detect a trash bag in frame-by-frame detection (due to occlusion,
    low contrast, or small size), CSRT takes over from the last known bounding box.
    """
    def __init__(self, trash_bag_class_id=1):
        self.trash_bag_class_id = trash_bag_class_id
        self.csrt_tracker = None
        self.tracking = False
        self.last_known_box = None  # [x1, y1, x2, y2]
        self.missed_frames = 0
        self.max_missed_frames = 30  # Stop CSRT tracking if YOLO fails to re-detect after N frames

    def update(self, frame, yolo_detections):
        """
        Args:
            frame: numpy array (H, W, 3)
            yolo_detections: list of dicts or objects with 'class_id', 'xyxy', 'confidence'
        Returns:
            tuple: (bbox [x1, y1, x2, y2], is_fallback_flag, confidence_score)
        """
        trash_bag_detected = False
        best_det_box = None
        best_conf = 0.0

        # Search for trash bag in current YOLO detections
        for det in yolo_detections:
            class_id = det.get('class_id') if isinstance(det, dict) else getattr(det, 'class_id', None)
            if class_id == self.trash_bag_class_id:
                trash_bag_detected = True
                conf = det.get('confidence', 1.0) if isinstance(det, dict) else getattr(det, 'confidence', 1.0)
                if conf > best_conf:
                    best_conf = conf
                    best_det_box = det.get('xyxy') if isinstance(det, dict) else getattr(det, 'xyxy', None)

        if trash_bag_detected and best_det_box is not None:
            # YOLO detected the bag: update last known box and reset/re-initialize CSRT tracker
            self.last_known_box = [int(v) for v in best_det_box]
            self.missed_frames = 0
            self.tracking = False  # Reset CSRT state to prefer fresh YOLO detection
            return self.last_known_box, False, best_conf

        # If YOLO did NOT detect the bag in this frame:
        if not trash_bag_detected and self.last_known_box is not None:
            x1, y1, x2, y2 = self.last_known_box
            w, h = x2 - x1, y2 - y1

            if w > 5 and h > 5:  # Ensure valid bounding box dimensions
                if not self.tracking:
                    # Initialize CSRT tracker with OpenCV
                    if hasattr(cv2, 'TrackerCSRT_create'):
                        self.csrt_tracker = cv2.TrackerCSRT_create()
                    else:
                        # Support older/newer opencv-contrib-python versions
                        self.csrt_tracker = cv2.legacy.TrackerCSRT_create()
                    
                    bbox_csrt = (x1, y1, w, h)
                    self.csrt_tracker.init(frame, bbox_csrt)
                    self.tracking = True

                if self.tracking:
                    self.missed_frames += 1
                    if self.missed_frames <= self.max_missed_frames:
                        success, bbox = self.csrt_tracker.update(frame)
                        if success:
                            x, y, w, h = [int(v) for v in bbox]
                            fallback_box = [x, y, x + w, y + h]
                            self.last_known_box = fallback_box
                            return fallback_box, True, 0.5  # Return CSRT predicted bounding box

        # Reset tracking if expired or lost
        self.tracking = False
        return None, False, 0.0
