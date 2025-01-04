import cv2
import numpy as np
from insightface.app import FaceAnalysis
from yaml import warnings

from classifier_models.Convnext import Net
import torch
import torch.nn.functional as F

# 人脸情绪识别数据集的标签
data_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_face_expression(face_crop):
    model = Net(num_classes=7).to(device)
    model.eval()
    model.load_state_dict(torch.load('./train_result/model_5.pth'))
    input_data = torch.tensor(face_crop, dtype=torch.float32).to(device)
    output = model(input_data)

    output_acc, output_label= torch.max(F.softmax(output, dim=1), dim=1)

    return output_label.item(), output_acc.item()

class Face_Detection:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l', root= './insight_face', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def run(self, frame):
        h, w = frame.shape[:2]
        faces = self.app.get(frame)
        crop_frame = []
        if len(faces) != 0:
            for face in faces:
                box = face.bbox.astype('int')
                box[0::2] = np.clip(box[0::2], 0, w)  # 限制 box[0] 和 box[2] 在 [0, w]
                box[1::2] = np.clip(box[1::2], 0, h)  # 限制 box[1] 和 box[3] 在 [0, h]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                gray_face = cv2.cvtColor(frame[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (48, 48))
                gray_face = np.expand_dims(gray_face, axis=0)

                crop_frame.append(gray_face)

            return frame, crop_frame, box
        else:
            return frame, None, None




def main():
    face_detection = Face_Detection()
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 读取摄像头的画面
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, crop_frame, box = face_detection.run(frame)

        if crop_frame is not None:
            output_label, output_acc = get_face_expression(crop_frame)
            emojis_img = cv2.imread(f"./emojis/{data_labels[output_label]}.png")
            emojis_img = cv2.resize(emojis_img, (box[2]-box[0], box[3]-box[1]))
            print(f'box: {box}')
            frame[box[1]:box[3], box[0]:box[2]] = emojis_img
            print(f'output_label: {data_labels[output_label]}, output_acc: {output_acc*100:.2f}%')

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()