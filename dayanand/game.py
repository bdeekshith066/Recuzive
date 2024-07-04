import math
import random
import cv2
import numpy as np
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import cvzone

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

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

def app():
    st.title("Snake Game with Hand Tracking")

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    game = SnakeGameClass("Donut.png")

    with st.form("game_form"):
        start_button = st.form_submit_button("Start Game")

        if start_button:
            FRAME_WINDOW = st.image([])

            while True:
                success, img = cap.read()
                if not success:
                    break
                img = cv2.flip(img, 1)
                hands, img = detector.findHands(img, flipType=False)

                if hands:
                    lmList = hands[0]['lmList']
                    pointIndex = lmList[8][0:2]
                    img = game.update(img, pointIndex)

                FRAME_WINDOW.image(img, channels="BGR", use_column_width=True)

            cap.release()
            cv2.destroyAllWindows()

# Run the app
if __name__ == "__main__":
    app()
