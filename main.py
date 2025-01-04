import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

class Face_Detection:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l', root= './insight_face', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def run(self, frame):
        faces = self.app.get(frame)
        for face in faces:
            box = face.bbox.astype('int')
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        return frame


if __name__ == "__main__":
    face_detection = Face_Detection()
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 读取摄像头的画面
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = face_detection.run(frame)

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break