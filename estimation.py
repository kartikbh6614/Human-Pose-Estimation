#pip install mediapipe

import cv2
import mediapipe as mp

# Step 2: Initialize MediaPipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Step 3: Load an image
image = cv2.imread('sample_image.jpg')

# Step 4: Process the image to estimate keypoints
results = pose.process(image)

# Step 5: Interpret and use the results
if results.pose_landmarks:
    for landmark in results.pose_landmarks.landmark:
        x, y, z = landmark.x, landmark.y, landmark.z
        print(f"Keypoint: x={x}, y={y}, z={z}")

# Display the image with keypoints (optional)
for landmark in results.pose_landmarks.landmark:
    h, w, c = image.shape
    cx, cy = int(landmark.x * w), int(landmark.y * h)
    cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

