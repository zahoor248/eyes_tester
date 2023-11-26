import cv2
import dlib
from imutils import face_utils
from scipy.spatial.distance import euclidean

# Initialize Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")  # Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Calculate the Euclidean distances between the two sets of vertical eye landmarks
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])

    # Calculate the Euclidean distance between the horizontal eye landmarks
    C = euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Set the threshold for eye aspect ratio to determine if eyes are open or closed
EAR_THRESHOLD = 0.3

# Open a video capture stream (use 0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Determine the facial landmarks for the face
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate the eye aspect ratio for each eye
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)

        # Compute the average eye aspect ratio
        ear_avg = (ear_left + ear_right) / 2.0

        # Check if the average eye aspect ratio is below the threshold
        if ear_avg < EAR_THRESHOLD:
            cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw the facial landmarks on the frame
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("Eye Detection", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
