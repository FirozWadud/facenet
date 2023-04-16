import typing
import numpy as np
import av
from tqdm import tqdm


class Engine:

    def __init__(
            self,
            url: str = "",
            show: bool = False,
            flip_view: bool = False,
            custom_objects: typing.Iterable = [],
            start_video_frame: int = 0,
            end_video_frame: int = 0,
            break_on_end: bool = False,
    ) -> None:
        self.url = url
        self.show = show
        self.flip_view = flip_view
        self.custom_objects = custom_objects
        self.start_video_frame = start_video_frame
        self.end_video_frame = end_video_frame
        self.break_on_end = break_on_end

    def flip(self, frame: np.ndarray) -> np.ndarray:

        if self.flip_view:
            return np.flip(frame, axis=1)

        return frame

    def custom_processing(self, frame: np.ndarray) -> np.ndarray:

        if self.custom_objects:
            for custom_object in self.custom_objects:
                frame = custom_object(frame)

        return frame

    def display(self, frame: np.ndarray, waitTime: int = 1) -> bool:

        if self.show:
            cv2.imshow('FaceNet Version recognition', frame)
            k = cv2.waitKey(waitTime)
            if k & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return False

        return True

    def process_video(self):
        """Process video stream for given video_path
        """
        container = av.open(self.url)
        stream = container.streams.video[0]

        for packet in container.demux(stream):
            for frame in packet.decode():
                if self.start_video_frame > 0:
                    self.start_video_frame -= 1
                    continue

                if self.end_video_frame > 0 and packet.pts >= self.end_video_frame:
                    break

                frame = frame.to_rgb().to_ndarray()
                frame = self.custom_processing(self.flip(frame))

                if not self.display(frame):
                    break

        else:
            if self.break_on_end:
                raise StopIteration("End of video")

        container.close()

    def process_image(self, image_path: str):
        """Process single image for given image_path
        """
        frame = cv2.imread(image_path)
        frame = self.custom_processing(self.flip(frame))
        self.display(frame, waitTime=0)

    def run(self):
        """Main object function to start processing image, video or webcam input
        """
        if self.url:
            self.process_video()
        else:
            self.process_webcam()
