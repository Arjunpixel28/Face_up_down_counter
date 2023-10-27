# Imports
import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection module
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Video Capture
cap = cv2.VideoCapture(0)

# Dictionary to store face states (up, down)
face_states = {}

# Counter for upward and downward movements
up_count = 0
down_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results_detection = face_detection.process(rgb_frame)

    if results_detection.detections:
        for idx, detection in enumerate(results_detection.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_id = idx  # Use the index as a unique ID for each detected face

            # Set initial head position and state for each face
            if face_id not in face_states:
                face_states[face_id] = {
                    'head_position': int(bboxC.ymin * ih) + int(bboxC.height * ih / 2),
                    'state': None
                }

            nose_y = int(bboxC.ymin * ih) + int(bboxC.height * ih / 2)
            
            # Determine head state (up or down) based on the change in nose position
            if nose_y < face_states[face_id]['head_position'] - 5:
                if face_states[face_id]['state'] != 'up':
                    face_states[face_id]['state'] = 'up'
                    up_count += 1  # Increment the upward movement counter
            elif nose_y > face_states[face_id]['head_position'] + 5:
                if face_states[face_id]['state'] != 'down':
                    face_states[face_id]['state'] = 'down'
                    down_count += 1  # Increment the downward movement counter
            face_states[face_id]['head_position'] = nose_y  # Update head position for the next iteration

            # Draw bounding box and keypoints on the frame
            mp_drawing.draw_detection(frame, detection)
            cv2.putText(frame, f'Face {int(face_id)} State: {face_states[face_id]["state"]}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes and face states
    cv2.putText(frame, f'Upward Count: {up_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Downward Count: {down_count}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
