from flask import Flask, request, jsonify
import cv2
import dlib
from imutils import face_utils
from scipy.spatial.distance import euclidean
import numpy as np

app = Flask(__name__)

# Initialize Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.3

@app.route('/eye_detection', methods=['POST'])
def eye_detection():
    # Receive image file
    file = request.files['image']
    # Convert the image file to a NumPy array
    image_np = np.frombuffer(file.read(), np.uint8)
    # Decode the NumPy array into an OpenCV image
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
  
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)

        ear_avg = (ear_left + ear_right) / 2.0

        if ear_avg < EAR_THRESHOLD:
            cv2.putText(img, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    # Save the result image (optional)
    cv2.imwrite("result.jpg", img)

    return jsonify({"status": "success", "message": "Eye detection endpoint"})

if __name__ == '__main__':
    app.run(debug=True)
