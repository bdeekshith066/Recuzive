import cv2
import mediapipe as mp
import streamlit as st
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import cvzone

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to count fingers
def count_fingers(hand_landmarks):
    if hand_landmarks:
        fingers = [
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
        ]
        
        finger_count = sum(fingers)
        
        # Detect reverse numbers (e.g., "2" shown with index and middle finger touching the thumb)
        reverse_two = (
            fingers[0] and fingers[1] and
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
        )
        
        return finger_count, reverse_two
    return 0, False

# Function to check if hand is oriented correctly
def is_hand_oriented_correctly(hand_landmarks):
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    middle_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    return middle_finger_tip_y < wrist_y

# Function to run the finger counting detection and store metrics
def run_detection():
    cap = cv2.VideoCapture(0)
    right_hand_done = False
    left_hand_done = False

    FRAME_WINDOW = st.image([])

    # Metrics storage
    metrics = {
        "right_hand": [],
        "left_hand": []
    }

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture image from camera.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    if is_hand_oriented_correctly(hand_landmarks):
                        finger_count, reverse_two = count_fingers(hand_landmarks)
                        if not right_hand_done and hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                            cv2.putText(image, f'Right Hand Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            if reverse_two:
                                cv2.putText(image, 'Reverse 2 detected on right hand!', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            if finger_count == 5:
                                right_hand_done = True
                                cv2.putText(image, 'Good job! Right hand finger counting complete.', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                FRAME_WINDOW.image(image, channels="BGR")
                                st.write("Good job! Right hand finger counting complete.")
                                st.write("Please show your left hand for finger counting.")
                                cv2.waitKey(2000)
                                metrics["right_hand"].append(finger_count)

                        if right_hand_done and not left_hand_done and hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x > 0.5:
                            cv2.putText(image, f'Left Hand Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            if reverse_two:
                                cv2.putText(image, 'Reverse 2 detected on left hand!', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            if finger_count == 5:
                                left_hand_done = True
                                cv2.putText(image, 'Good job! Left hand finger counting complete.', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                FRAME_WINDOW.image(image, channels="BGR")
                                st.write("Good job! Left hand finger counting complete.")
                                metrics["left_hand"].append(finger_count)
                                cap.release()
                                cv2.destroyAllWindows()
                                return metrics
                    else:
                        cv2.putText(image, 'Orient your hand upright', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            FRAME_WINDOW.image(image, channels="BGR")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return metrics

# Snake game class
class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []
        self.lengths = []  # distance between each point
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = (0, 0)
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        if self.imgFood is None:
            raise ValueError(f"Could not read image at {pathFood}")
        if self.imgFood.shape[2] == 3:  # If the image does not have an alpha channel, add one
            self.imgFood = cv2.cvtColor(self.imgFood, cv2.COLOR_BGR2BGRA)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = (0, 0)
        self.score = 0
        self.gameOver = False
        self.randomFoodLocation()
        
    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)
        
    def update(self, imgMain, currentHead):
        if self.gameOver:
            st.error("Game Over")
            st.write(f'Your Score: {self.score}')
            return imgMain

        px, py = self.previousHead
        cx, cy = currentHead

        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        # Length Reduction
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths.pop(i)
                self.points.pop(i)
                if self.currentLength < self.allowedLength:
                    break
       
        # Draw Snake
        if self.points:
            for i, point in enumerate(self.points):
                if i != 0:
                    cv2.line(imgMain, tuple(self.points[i - 1]), tuple(self.points[i]), (0, 0, 255), 20)
            cv2.circle(imgMain, tuple(self.points[-1]), 20, (0, 255, 0), cv2.FILLED)

        # Draw Food
        rx, ry = self.foodPoint
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

        # Check if snake ate the Food
        if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                ry - self.hFood // 2 < cy < ry + self.hFood // 2:
            self.randomFoodLocation()
            self.allowedLength += 50
            self.score += 1
            st.success(f"Score: {self.score}")

        return imgMain

# Snake game runner function
def run_snake_game():
    st.title("Snake Game with Hand Tracking")

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Initialize hand detector here
    detector = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)  
    game = SnakeGameClass("Donut.png")

    FRAME_WINDOW = st.image([])

    # Metrics storage
    snake_metrics = {
        "score": [],
        "food_positions": [],
        "snake_length": []
    }

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                pointIndex = [int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 1280), 
                              int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 720)]
                img = game.update(img, pointIndex)

                # Store metrics
                snake_metrics["score"].append(game.score)
                snake_metrics["food_positions"].append(game.foodPoint)
                snake_metrics["snake_length"].append(game.allowedLength)

        FRAME_WINDOW.image(img, channels="BGR", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()
    return snake_metrics

# Main app function
def app():
    gradient_text_html = """
    <style>
    .gradient-text {
        font-weight: bold;
        background: -webkit-linear-gradient(left, #07539e, #4fc3f7, #ffffff);
        background: linear-gradient(to right, #07539e, #4fc3f7, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-size: 3.2em;
    }
    </style>
    <div class="gradient-text">Physio</div>
    """

    # Render the gradient text
    st.markdown(gradient_text_html, unsafe_allow_html=True)
    st.write('Empowering movement and enhancing life')
    st.image('divider.png')

    col1, col2, col3, col4, col5 = st.columns([0.05, 0.6, 0.1, 0.48, 0.1])

    with col2:
        with st.form("Finger"):
            st.image('fingerr.jpg')
            finger_button = st.form_submit_button("Finger")

            if finger_button:
                finger_metrics = run_detection()
                st.write(finger_metrics)

                # Visualize Finger Metrics
                st.header("Finger Counting Metrics")
                st.write("Right Hand Metrics")
                fig, ax = plt.subplots()
                ax.plot(finger_metrics["right_hand"], marker='o')
                ax.set_xlabel("Count")
                ax.set_ylabel("Finger Count")
                st.pyplot(fig)

                st.write("Left Hand Metrics")
                fig, ax = plt.subplots()
                ax.plot(finger_metrics["left_hand"], marker='o')
                ax.set_xlabel("Count")
                ax.set_ylabel("Finger Count")
                st.pyplot(fig)

                # Download Finger Metrics
                finger_df = pd.DataFrame({
                    "right_hand": finger_metrics["right_hand"],
                    "left_hand": finger_metrics["left_hand"]
                })
                csv = finger_df.to_csv(index=False)
                

    with col4:
        with st.form("gamify"):
            st.image('sukha.jpg')
            gamify_button = st.form_submit_button("Gamify - Physio")

            if gamify_button:
                snake_metrics = run_snake_game()
                st.write(snake_metrics)

                # Visualize Snake Game Metrics
                st.header("Snake Game Metrics")
                st.write("Score Progression")
                fig, ax = plt.subplots()
                ax.plot(snake_metrics["score"], marker='o')
                ax.set_xlabel("Frame")
                ax.set_ylabel("Score")
                st.pyplot(fig)

                st.write("Snake Length Progression")
                fig, ax = plt.subplots()
                ax.plot(snake_metrics["snake_length"], marker='o')
                ax.set_xlabel("Frame")
                ax.set_ylabel("Snake Length")
                st.pyplot(fig)

                # Bar Chart for Food Positions
                st.write("Food Positions Bar Chart")
                food_positions = snake_metrics["food_positions"]
                x_positions = [pos[0] for pos in food_positions]
                y_positions = [pos[1] for pos in food_positions]
                fig, ax = plt.subplots()
                ax.bar(range(len(food_positions)), x_positions, label='X Positions')
                ax.bar(range(len(food_positions)), y_positions, label='Y Positions', alpha=0.5)
                ax.set_xlabel("Food Item")
                ax.set_ylabel("Position")
                ax.legend()
                st.pyplot(fig)

                # Pie Chart for Score Distribution
                st.write("Score Distribution Pie Chart")
                score_counts = {i: snake_metrics["score"].count(i) for i in set(snake_metrics["score"])}
                fig, ax = plt.subplots()
                ax.pie(score_counts.values(), labels=score_counts.keys(), autopct='%1.1f%%')
                ax.set_title("Score Distribution")
                st.pyplot(fig)

                # Download Snake Game Metrics
                snake_df = pd.DataFrame({
                    "score": snake_metrics["score"],
                    "food_positions": snake_metrics["food_positions"],
                    "snake_length": snake_metrics["snake_length"]
                })
                csv = snake_df.to_csv(index=False)
                st.form_submit_button(
                    label="Download Snake Game Metrics as CSV",
                    data=csv,
                    file_name='snake_game_metrics.csv',
                    mime='text/csv',
                )

if __name__ == "__main__":
    app()
