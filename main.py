import cv2
import numpy as np
import dlib
import time
from values import FACE_CASCADE_XML_FILE, DLIB_SHAPE_PREDICTOR_FILE
from age_gender_detection import AgeGenderDetection
from blinking_detection import BlinkingDetection
from utils import Utils


face_cascade = cv2.CascadeClassifier(FACE_CASCADE_XML_FILE)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR_FILE)
FONT = cv2.FONT_HERSHEY_PLAIN
MAIN_FRAME = "EYE TRACKING"

video = cv2.VideoCapture(0)


global_faces = []

while True:
    start = time.time()
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)
    AgeGenderDetection.detect_age_gender(frame)

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_box = face_cascade.detectMultiScale(grayImage)
    Utils.detect_faces(face_box, frame)

    faces = detector(grayImage)
    height, width, _ = frame.shape
    right_mask = np.zeros((height, width), np.uint8)
    left_mask = np.zeros((height, width), np.uint8)
    global_faces = Utils.global_face_control(global_faces, faces)
    result = 0

    for i, face in enumerate(faces):

        context = BlinkingDetection.detect_blink(face, predictor, grayImage, right_mask, left_mask)

        Utils.draw_eye(context, ['left', 'right'], frame, MAIN_FRAME)

        if context['is_blinking']:
            cv2.putText(frame, "BLİNKİNG", (50, 150), FONT, 7, (255, 0, 0))
        else:
            end = time.time()
            result = end - start
            global_faces[i]['time'] += result
            print(f"Yüz-{global_faces[i]['index']} Bakış Süresi:  ", global_faces[i]['time'])

    else:
        end = time.time()

    cv2.imshow(MAIN_FRAME, frame)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break

print("--"*50)
print("BAKMA SÜRELERİ: ", global_faces)
video.release()
cv2.destroyAllWindows()
