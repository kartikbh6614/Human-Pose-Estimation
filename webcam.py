import cv2
import mediapipe as mp

# Step 2: Initialize MediaPipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2)

# Step 3: Open a webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Step 4: Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        continue

    # Process the frame to estimate keypoints
    results = pose.process(frame)

    # Step 5: Interpret and use the results
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x, y, z = landmark.x, landmark.y, landmark.z
            print(f"Keypoint: x={x}, y={y}, z={z}")

    # Display the frame with keypoints (optional)
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, c = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
