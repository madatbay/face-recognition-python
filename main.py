import cv2
import os
import pickle

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = {}
with open('label.pickle', 'rb') as file:
    labels = pickle.load(file)
    labels = {v: k for k, v in labels.items()}

capture = cv2.VideoCapture(0)


def change_resolution(frame, width, height):
    frame.set(3, width)
    frame.set(4, height)


def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


filename = 'video.avi'
frames_per_second = 24.0
res = '720p'

STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


def get_dims(capture, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_resolution(capture, width, height)
    return width, height


VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

# Capture video obj
# out = cv2.VideoWriter(filename, get_video_type(
#     filename), 25, get_dims(capture, res))


while True:
    successful_frame, frame = capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coords = trained_face_data.detectMultiScale(gray_frame)
    for (x, y, w, h) in face_coords:
        roi_gray = gray_frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 60 and conf <= 80:
            cv2.putText(
                frame, labels[id_], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Face Detector', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
