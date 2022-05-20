import cv2
import numpy as np
import dlib
from math import hypot
import time


def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
      confidence=detection[0,0,i,2]
      if confidence>0.7:
        x1=int(detection[0, 0, i, 3] * frameWidth)
        y1=int(detection[0, 0, i, 4] * frameHeight)
        x2=int(detection[0, 0, i, 5] * frameWidth)
        y2=int(detection[0, 0, i, 6] * frameHeight)
        bboxs.append([x1,y1,x2,y2])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
    return frame,bboxs


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-13)', '(15-18)', '(19-25)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video=cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
padding=20

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def gaze_dedection(mask,gray,eye_region):
    cv2.polylines(mask,[eye_region],True,255,2)
    cv2.fillPoly(mask,[eye_region],255)
    eye=cv2.bitwise_and(gray,gray,mask=mask)
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    result_eye = cv2.resize(gray_eye, None, fx=5, fy=5)
    mid_point = [(max_y-min_y)/2, (max_x - min_x)/2]
    return eye,threshold_eye,result_eye, mid_point

is_blinking=False
start = time.time()

while True:
    ret,frame=video.read()
    frameNet,bboxs=faceBox(faceNet,frame)
    for bbox in bboxs:
       # face=frameNet[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)
        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]

        label="{},{}".format(gender,age)
        cv2.rectangle(frame,(bbox[0],bbox[1]-30),(bbox[2],bbox[1]),(0,255,0),-1)
        cv2.putText(frame,label,(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImage)
    if len(faces) == 0:
        cv2.putText(frame, "Number of faces detected: 0", (0, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)
    else:
        cv2.putText(frame, "Number of faces detected: " + str(faces.shape[0]), (0, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Image with faces', frame)

    faces = detector(grayImage)
    height, width, _ = frame.shape
    right_mask = np.zeros((height, width), np.uint8)
    left_mask = np.zeros((height, width), np.uint8)
    times = []
    is_face = False
    result = 0

    for face in faces:
        blink_time = time.time()
        is_face = True
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        landmarks = predictor(grayImage, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Gaze detection
        right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                     (landmarks.part(43).x, landmarks.part(43).y),
                                     (landmarks.part(44).x, landmarks.part(44).y),
                                     (landmarks.part(45).x, landmarks.part(45).y),
                                     (landmarks.part(46).x, landmarks.part(46).y),
                                     (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        right_eye, threshold_eye1, eye1, right_mid_point = gaze_dedection(right_mask, grayImage, right_eye_region)

        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        left_eye, threshold_eye2, eye2, left_mid_point = gaze_dedection(left_mask, grayImage, left_eye_region)


        cv2.imshow("Eye", eye1)
        cv2.imshow("Threshold", threshold_eye1)
        cv2.imshow("Right eye", right_eye)

        cv2.imshow("Eye2", eye2)
        cv2.imshow("Threshold2", threshold_eye2)
        cv2.imshow("Left eye", left_eye)

        if blinking_ratio > 5.7:
            is_blinking = True
            cv2.putText(frame, "BLİNKİNG", (50, 150), font, 7, (255, 0, 0))
            result -= time.time() - blink_time

        end = time.time()

        if is_face:
            result = end - start
            print("bakis suresi ", result)

    else:
        is_face = False
        end = time.time()

    cv2.imshow("Frame", frame)
    k=cv2.waitKey(1)

    if k == ord('q'):
        break



video.release()
cv2.destroyAllWindows()
