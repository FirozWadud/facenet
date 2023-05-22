import cv2
import typing
import numpy as np
from mtcnn import MTCNN

class MTCNNFaceDetection:
    def __init__(
        self,
        min_confidence: float = 0.5,
        draw_detections: bool = True,
        color: typing.Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
    ) -> None:
        self.min_confidence = min_confidence
        self.draw_detections = draw_detections
        self.color = color
        self.thickness = thickness
        self.detector = MTCNN()

    def tlbr(self, frame: np.ndarray, mtcnn_detections: typing.List) -> np.ndarray:
        detections = []
        for detection in mtcnn_detections:
            x, y, width, height = detection['box']
            left, top, right, bottom = x, y, x + width, y + height
            detections.append([top, left, bottom, right])

        return np.array(detections)

    def __call__(self, frame: np.ndarray, return_tlbr: bool = False) -> np.ndarray:
        mtcnn_detections = self.detector.detect_faces(frame)

        if return_tlbr:
            if mtcnn_detections:
                return self.tlbr(frame, mtcnn_detections)
            return []

        if mtcnn_detections:
            if self.draw_detections:
                for detection in mtcnn_detections:
                    x, y, width, height = detection['box']
                    cv2.rectangle(frame, (x, y), (x + width, y + height), self.color, self.thickness)
        return frame
