import cv2
from values import *


class AgeGenderDetection:

    @staticmethod
    def detect_face(face_net, frame):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
        face_net.setInput(blob)
        detection = face_net.forward()
        b_boxs = []
        for i in range(detection.shape[2]):
            confidence = detection[0, 0, i, 2]
            if confidence > 0.7:
                x1 = int(detection[0, 0, i, 3] * frame_width)
                y1 = int(detection[0, 0, i, 4] * frame_height)
                x2 = int(detection[0, 0, i, 5] * frame_width)
                y2 = int(detection[0, 0, i, 6] * frame_height)
                b_boxs.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return b_boxs

    @staticmethod
    def detect_age_gender(frame):
        face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        b_boxs = AgeGenderDetection.detect_face(face_net, frame)
        for b_box in b_boxs:
            # face=frameNet[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            face = frame[max(0, b_box[1] - PADDING):min(b_box[3] + PADDING, frame.shape[0] - 1),
                         max(0, b_box[0] - PADDING):min(b_box[2] + PADDING, frame.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_pred = gender_net.forward()
            gender = GENDER_LIST[gender_pred[0].argmax()]

            age_net.setInput(blob)
            age_pred = age_net.forward()
            age = AGE_LIST[age_pred[0].argmax()]

            label = "{},{}".format(gender, age)
            cv2.rectangle(frame, (b_box[0], b_box[1] - 30), (b_box[2], b_box[1]), (0, 255, 0), -1)
            cv2.putText(frame, label, (b_box[0], b_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
