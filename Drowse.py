import cv2
import dlib
from scipy.spatial import distance
cap = cv2.VideoCapture(0)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def calculate_eye_aspect_ratio(eye_points):
    # Calculate the Euclidean distance between eye landmarks
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        left_ear = calculate_eye_aspect_ratio(left_eye_points)
        right_ear = calculate_eye_aspect_ratio(right_eye_points)
        threshold = 0.25
        if left_ear < threshold and right_ear < threshold:
            # Alert: Driver is drowsy
            print("Driver is drowsy!")
    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
