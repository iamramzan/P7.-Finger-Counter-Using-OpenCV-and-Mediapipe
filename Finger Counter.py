# P8. Finger Counter Using OpenCV and Mediapipe

# Importing necessary libraries
import cv2  # OpenCV for video capture and image processing
import mediapipe as mp  # Mediapipe for hand tracking and pose estimation

# Initializing Mediapipe's drawing utilities and holistic model
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks on images
mp_holistic = mp.solutions.holistic  # For holistic models (not used in this project)

# Initializing Mediapipe's hand module for hand detection and tracking
mp_hands = mp.solutions.hands

# Setting up webcam input
cap = cv2.VideoCapture(1)  # Open webcam (index 1, adjust based on your setup)

# Using Mediapipe Hands with minimum detection and tracking confidence levels
with mp_hands.Hands(
        min_detection_confidence=0.5,  # Confidence threshold for detecting hands
        min_tracking_confidence=0.5  # Confidence threshold for tracking hands
) as hands:
    while cap.isOpened():  # Loop until the webcam is closed
        success, image = cap.read()  # Read a frame from the webcam
        if not success:  # If no frame is captured, print a warning and continue
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view and convert BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Optimize performance by marking the image as not writeable (pass by reference)
        image.flags.writeable = False
        results = hands.process(image)  # Process the image to detect and track hands

        # Allow modifications to the image and convert it back to BGR for OpenCV display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get the image dimensions
        image_height, image_width, _ = image.shape

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Initialize a string to store the names of raised fingers
                fin = ''

                # Check if the index finger is raised
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > \
                   hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y:
                    val1 = 0  # Index finger is not raised
                else:
                    val1 = 1  # Index finger is raised
                    fin = 'Index '  # Add "Index" to the raised fingers string

                # Check if the middle finger is raised
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > \
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y:
                    val2 = 0  # Middle finger is not raised
                else:
                    val2 = 1  # Middle finger is raised
                    fin += 'Middle '  # Add "Middle" to the raised fingers string

                # Check if the ring finger is raised
                if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > \
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y:
                    val3 = 0  # Ring finger is not raised
                else:
                    val3 = 1  # Ring finger is raised
                    fin += 'Ring '  # Add "Ring" to the raised fingers string

                # Check if the pinky finger is raised
                if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > \
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y:
                    val4 = 0  # Pinky finger is not raised
                else:
                    val4 = 1  # Pinky finger is raised
                    fin += 'Pinky '  # Add "Pinky" to the raised fingers string

                # Calculate the total number of raised fingers
                val = val1 + val2 + val3 + val4

                # Display the total number of raised fingers on the image
                fps = str(val) + ' fingers'
                cv2.putText(image, fps, (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

                # Display the names of the raised fingers on the image
                cv2.putText(image, fin, (0, 60), cv2.FONT_HERSHEY_PLAIN, 2, (10, 10, 0), 2)

        # Show the annotated image in a window
        cv2.imshow('MediaPipe Hands', image)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
