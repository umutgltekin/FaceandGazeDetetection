from math import hypot
import numpy as np
import cv2


class Utils:

    @staticmethod
    def find_midpoint(p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    @staticmethod
    def get_blinking_ratio(eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = Utils.find_midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = Utils.find_midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
        hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        ratio = hor_line_lenght / ver_line_lenght
        return ratio

    @staticmethod
    def detect_faces(faces, frame):
        if len(faces) == 0:
            cv2.putText(frame, "Number of faces detected: 0", (0, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.putText(frame, "Number of faces detected: " + str(faces.shape[0]), (0, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

    @staticmethod
    def get_eye_region(landmarks, numbers):
        return np.array([(landmarks.part(numbers[0]).x, landmarks.part(numbers[0]).y),
                         (landmarks.part(numbers[1]).x, landmarks.part(numbers[1]).y),
                         (landmarks.part(numbers[2]).x, landmarks.part(numbers[2]).y),
                         (landmarks.part(numbers[3]).x, landmarks.part(numbers[3]).y),
                         (landmarks.part(numbers[4]).x, landmarks.part(numbers[4]).y),
                         (landmarks.part(numbers[5]).x, landmarks.part(numbers[5]).y)], np.int32)

    @staticmethod
    def draw_eye(context, keys, frame, frame_name):
        for key in keys:
            cv2.rectangle(frame, (context[key]['frame'][0], context[key]['frame'][1]),
                          (context[key]['frame'][2], context[key]['frame'][3]), (0, 255, 0), 2)
            cv2.imshow(frame_name, frame)

    @staticmethod
    def get_face_dict(i, face, time_value):
        if isinstance(face, list):
            start_x, start_y, end_x, end_y = face[0], face[1], face[2], face[3]
        else:
            start_x = face.left()
            start_y = face.top()
            end_x = face.right()
            end_y = face.bottom()
        return {"index": i, "face": [start_x, start_y, end_x, end_y], "time": time_value}

    @staticmethod
    def global_face_control(global_faces, faces):
        if len(global_faces) == 0:
            global_faces = [Utils.get_face_dict(i, face, 0) for i, face in enumerate(faces)]
        elif len(global_faces) > len(faces) > 0:
            tmp_faces = [g_face for g_face in global_faces]
            global_faces = []
            for face in faces:
                is_exist = False
                index = 0
                for i, t_face in enumerate(tmp_faces):
                    index = i
                    if abs(int(t_face["face"][0]) - int(face.left())) < 20:
                        is_exist = True
                        break
                if is_exist:
                    global_faces.append(tmp_faces[index])
                    del tmp_faces[index]

            for t_face in tmp_faces:
                global_faces.append(t_face)
        elif len(global_faces) < len(faces):
            for i, face in enumerate(faces[-len(global_faces):]):
                global_faces.append(Utils.get_face_dict(len(global_faces) + i, face, 0))

        return global_faces
