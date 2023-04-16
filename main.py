# main.py
from utils import FPSmetric
from engine1 import Engine
from faceDetection import MPFaceDetection
from faceNet.faceNet import FaceNet

if __name__ == '__main__':
    facenet = FaceNet(
        detector = MPFaceDetection(),
        onnx_model_path = "models/faceNet.onnx",
        anchors = "faces",
        force_cpu = True,
    )
    engine = Engine(webcam_id= "rtsp://admin:admin123@192.168.0.150:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif", show=True, custom_objects=[facenet, FPSmetric()])

    # save first face crop as anchor, otherwise don't use
    while not facenet.detect_save_faces(engine.process_webcam(return_frame=True), output_dir="faces"):
        continue

    engine.run()