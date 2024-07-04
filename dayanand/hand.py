import cv2
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import cvzone
import os

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize game assets
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Define image paths
image_paths = {
    "background": "Resources/Background.jpg",
    "game_over": "Resources/game_over.jpg",
    "ball": "Resources/ddd.jpg",
    "left_bat": "Resources/left.jpg",
    "right_bat": "Resources/hight.jpg"
}

# Load and check images
images = {}
for name, path in image_paths.items():
    if os.path.exists(path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            st.error(f"Failed to load image: {path}")
            st.stop()
        # Check if the image has an alpha channel
        if image.shape[2] == 3:
            # Add an alpha channel
            alpha_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
            image = np.concatenate((image, alpha_channel), axis=2)
        images[name] = image
    else:
        st.error(f"Image not found: {path}")
        st.stop()

# Resize imgBackground to match the webcam frame size
imgBackground = cv2.resize(images["background"], (1280, 720))

# Ensure imgBackground has 3 channels for blending
if imgBackground.shape[2] == 4:
    imgBackground = cv2.cvtColor(imgBackground, cv2.COLOR_BGRA2BGR)

# Initialize game variables
ballPos = [100, 100]
speedX = 25
speedY = 25
gameOver = False
score = [0, 0]

def update_game():
    global ballPos, speedX, speedY, gameOver, score, images

    success, img = cap.read()
    if not success:
        st.error("Failed to read from webcam.")
        st.stop()

    img = cv2.flip(img, 1)
    imgRaw = img.copy()
    hands, img = detector.findHands(img, flipType=False)

    # Ensure img and imgBackground have the same size and number of channels
    imgBackground_resized = cv2.resize(imgBackground, (img.shape[1], img.shape[0]))

    img = cv2.addWeighted(img, 0.2, imgBackground_resized, 0.8, 0)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = images["left_bat"].shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, images["left_bat"], (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, images["right_bat"], (1195, y1))
                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1

    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    if gameOver:
        img = images["game_over"]
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (200, 0, 200), 5)
        
    else:
        # Move the Ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        img = cvzone.overlayPNG(img, images["ball"], ballPos)

        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    # Ensure imgRaw has 3 channels
    if imgRaw.shape[2] == 4:
        imgRaw = cv2.cvtColor(imgRaw, cv2.COLOR_BGRA2BGR)

    return img

def app():
    st.title("Ping Pong Game with Hand Tracking")

    with st.form("game_form"):
        start_button = st.form_submit_button("Start Game")

        if start_button:
            FRAME_WINDOW = st.image([])

            while True:
                img = update_game()
                FRAME_WINDOW.image(img, channels="BGR", use_column_width=True)

                # Reset the game if 'r' key is pressed
                if cv2.waitKey(1) == ord('r'):
                    global ballPos, speedX, speedY, gameOver, score, images
                    ballPos = [100, 100]
                    speedX = 22
                    speedY = 22
                    gameOver = False
                    score = [0, 0]
                    images["game_over"] = cv2.imread("Resources/game_over.jpg")
                
                if gameOver:
                    cap.release()
                    cv2.destroyAllWindows()
                    break

# Run the app
if __name__ == "__main__":
    app()
