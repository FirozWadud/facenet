import cv2
import stow
import typing
import numpy as np
from tqdm import tqdm


class Engine:

    def __init__(
        self,
        image_path: str = "",
        video_path: str = "",
        webcam_id: int = 0,
        show: bool = False,
        flip_view: bool = False,
        custom_objects: typing.Iterable = [],
        output_extension: str = 'out',
        start_video_frame: int = 0,
        end_video_frame: int = 0,
        break_on_end: bool = False,
        ) -> None:
        self.video_path = video_path
        self.image_path = image_path
        self.webcam_id = webcam_id
        self.show = show
        self.flip_view = flip_view
        self.custom_objects = custom_objects
        self.output_extension = output_extension
        self.start_video_frame = start_video_frame
        self.end_video_frame = end_video_frame
        self.break_on_end = break_on_end

    def flip(self, frame: np.ndarray) -> np.ndarray:

        if self.flip_view:
            return cv2.flip(frame, 1)

        return frame

    def custom_processing(self, frame: np.ndarray) -> np.ndarray:

        if self.custom_objects:
            for custom_object in self.custom_objects:
                frame = custom_object(frame)

        return frame

    def display(self, frame: np.ndarray, webcam: bool = False, waitTime: int = 1) -> bool:

        if self.show:
            cv2.imshow('FaceNet Version recognition', frame)
            k = cv2.waitKey(waitTime)
            if k & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return False

        return True

    def process_webcam(self, return_frame: bool = False) -> typing.Union[None, np.ndarray]:
        """Process webcam stream for given webcam_id
        """
        # Create a VideoCapture object for given webcam_id
        cap = cv2.VideoCapture(self.webcam_id)
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                print("Ignoring empty camera frame.")
                continue

            if return_frame:
                break

            frame = self.custom_processing(self.flip(frame))

            if not self.display(frame, webcam=True):
                break

        else:
            raise Exception(f"Webcam with ID ({self.webcam_id}) can't be opened")

        cap.release()
        return frame

    def run(self):
        """Main object function to start processing image, video or webcam input
        """
        if self.video_path:
            self.process_video()
        elif self.image_path:
            self.process_image(self.image_path)
        else:
            self.process_webcam()