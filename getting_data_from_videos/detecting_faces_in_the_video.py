import os
import cv2
import datetime
from constants_for_diarized_speakers import CASCADE_FILE_PATH


def detect_and_save_faces(video_path, output_folder):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_FILE_PATH)
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("video ended")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for i, (x, y, w, h) in enumerate(faces):
            face_image = frame[y:y+h, x:x+w]
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            face_filename = os.path.join(output_folder, f"face_{i}_{timestamp}.jpg")
            cv2.imwrite(face_filename, face_image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
