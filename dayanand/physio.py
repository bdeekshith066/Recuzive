import streamlit as st
import speech_recognition as sr
import pandas as pd
import datetime
import random
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import math
import random
import numpy as np
import os
import io
import cvzone
import random
import base64
import json
from PIL import Image
from streamlit_autorefresh import st_autorefresh
import time
import time as tm
from gtts import gTTS
from googletrans import Translator
import os


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

    #img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    return img

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to count fingers


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

# Function to recognize speech using the microphone
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        response = recognizer.recognize_google(audio)
    except sr.RequestError:
        response = "API unavailable"
    except sr.UnknownValueError:
        response = "Unable to recognize speech"
    return response

# Initialize recognizer and microphone
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Sample paragraphs
paragraphs = [
    "Exercising daily is good for your body and mind. Simple activities like walking or stretching can boost your mood. Exercise can also help improve your strength and balance.",
    "The beauty of the natural world unfolded before my eyes, as sunlight filtered through the lush canopy of trees. Each leaf shimmered with a verdant hue, while the gentle rustle of branches whispered secrets of the forest. In the embrace of natures splendor, I found solace and renewal, a sanctuary from the hustle and bustle of daily life.",
]

# Initialize session state for storing progress
if 'progress' not in st.session_state:
    st.session_state.progress = []

# Initialize session state for paragraph selection
if 'paragraph' not in st.session_state:
    st.session_state.paragraph = random.choice(paragraphs)


translator = Translator()

# Define the feedback text
feedback_text = "Ask the patient to speak. Observe him carefully while they speak and the scores will be displayed on the screen. Assess him with care regards NueroWell"

    # List of languages
languages = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn",
        "Telugu": "te",
        "Tamil": "ta",
        "Malayalam": "ml",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Bengali": "bn",
        "Punjabi": "pa"
    }

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
              font-size: 2.9em;
          }
          </style>
          <div class="gradient-text">Neuro Well Test Analysis</div>
          """

      # Render the gradient text
    st.markdown(gradient_text_html, unsafe_allow_html=True)
    
    st.image('divider.png')


    

    # Create columns
    col11, col22 = st.columns(2)

    # Language selection in column 2
    with col22:
        selected_language = st.selectbox("Choose a language for instructions", list(languages.keys()))

    # Translate the text
    if selected_language != "English":
        translated_text = translator.translate(feedback_text, dest=languages[selected_language]).text
    else:
        translated_text = feedback_text

    # Convert text to speech
    tts = gTTS(translated_text, lang=languages[selected_language])
    audio_file = "feedback.mp3"
    tts.save(audio_file)

    # Display the audio in column 1
    with col11:
        st.write('')
        st.audio(audio_file)

    # Remove the audio file after playing
    os.remove(audio_file)

    col8, col9 = st.columns([3, 1.5])

    with col8:
        
        with st.form("speech"):
            # Display the paragraph
            st.write("Please read the following paragraph aloud:")
            st.write(st.session_state.paragraph)

            # Button to start speech recognition
            if st.form_submit_button('Start Speech Recognition'):
                with st.spinner("Listening..."):
                    response = recognize_speech_from_mic(recognizer, microphone)
                st.write(f"Recognized Speech: {response}")

                # Get the current timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Calculate score based on the recognized speech
                original_words = set(st.session_state.paragraph.split())
                recognized_words = set(response.split())
                matched_words = original_words.intersection(recognized_words)
                score = len(matched_words)

                # Store the response and score in session state
                st.session_state.progress.append({
                    'timestamp': timestamp,
                    'response': response,
                    'score': score,
                    'total_words': len(original_words),
                    'matched_words': len(matched_words),
                    'unmatched_words': len(original_words) - len(matched_words)
                })

                # Save progress to a CSV file (or database)
                df = pd.DataFrame(st.session_state.progress)
                df.to_csv("progress.csv", index=False)

            # Display progress as a table
            if st.session_state.progress:
                df = pd.DataFrame(st.session_state.progress)
                st.write("Speech Recognition Progress")
                st.dataframe(df)

            st.write('')

    with col9:
        # Check if there is progress data to display graphs
        if st.session_state.progress:
            df = pd.DataFrame(st.session_state.progress)

            # Select box for graph type
            graph_choice = st.selectbox("Select Graph", ["Pie Chart", "Stacked Bar Chart", "Area Chart"])

            # Display the selected graph
            if graph_choice == "Area Chart":
                st.write("Progress Over Time (Area Chart)")
                plt.figure(figsize=(8, 4))
                plt.fill_between(df['timestamp'], df['score'], color="skyblue", alpha=0.4)
                plt.plot(df['timestamp'], df['score'], color="Slateblue", alpha=0.6)
                plt.xlabel('Timestamp')
                plt.ylabel('Score')
                plt.title('Speech Recognition Progress')
                plt.xticks(rotation=0)
                plt.grid(True)
                st.pyplot(plt)


            elif graph_choice == "Stacked Bar Chart":
                st.write("Scores (Stacked Bar Chart)")
                df.set_index('timestamp', inplace=True)
                df[['matched_words', 'unmatched_words']].plot(kind='bar', stacked=True, figsize=(8, 4), color=['#66b3ff', '#ff9999'])
                plt.xlabel('Timestamp')
                plt.ylabel('Count')
                plt.title('Matched vs Unmatched Words')
                plt.xticks(rotation=0)
                plt.grid(axis='y')
                st.pyplot(plt)

            elif graph_choice == "Pie Chart":
                st.write("Word Recognition (Pie Chart)")
                total_words = df['total_words'].sum()
                matched_words = df['matched_words'].sum()
                unmatched_words = total_words - matched_words
                labels = ['Matched Words', 'Unmatched Words']
                sizes = [matched_words, unmatched_words]
                colors = ['#ff9999', '#66b3ff']
                explode = (0.1, 0)  # explode 1st slice
                plt.figure(figsize=(8, 4))
                plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                        shadow=True, startangle=140)
                plt.title('Matched vs Unmatched Words')
                st.pyplot(plt)

            st.write('')

    st.image('divider.png')
    st.image('divider.png')
    

    
    with st.form("game"):
        vDrive = os.path.splitdrive(os.getcwd())[0]
        if vDrive == "C:": vpth = "C:/Users/Shawn/dev/utils/pixmatch/"   # local developer's disc
        else: vpth = "./"

        sbe = """<span style='font-size: 140px;
                            border-radius: 7px;
                            text-align: center;
                            display:inline;
                            padding-top: 3px;
                            padding-bottom: 3px;
                            padding-left: 0.4em;
                            padding-right: 0.4em;
                            '>
                            |fill_variable|
                            </span>"""

        pressed_emoji = """<span style='font-size: 24px;
                                        border-radius: 7px;
                                        text-align: center;
                                        display:inline;
                                        padding-top: 3px;
                                        padding-bottom: 3px;
                                        padding-left: 0.2em;
                                        padding-right: 0.2em;
                                        '>
                                        |fill_variable|
                                        </span>"""

        horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #635985;'><br>"    # thin divider line
        purple_btn_colour = """
                                <style>
                                    div.stButton > button:first-child {background-color: #4b0082; color:#ffffff;}
                                    div.stButton > button:hover {background-color: RGB(0,112,192); color:#ffffff;}
                                    div.stButton > button:focus {background-color: RGB(47,117,181); color:#ffffff;}
                                </style>
                            """

        mystate = st.session_state
        if "expired_cells" not in mystate: mystate.expired_cells = []
        if "myscore" not in mystate: mystate.myscore = 0
        if "plyrbtns" not in mystate: mystate.plyrbtns = {}
        if "sidebar_emoji" not in mystate: mystate.sidebar_emoji = ''
        if "emoji_bank" not in mystate: mystate.emoji_bank = []
        if "GameDetails" not in mystate: mystate.GameDetails = ['Medium', 6, 7, '']  # difficulty level, sec interval for autogen, total_cells_per_row_or_col, player name

        # common functions
        def ReduceGapFromPageTop(wch_section = 'main page'):
            if wch_section == 'main page': st.markdown(" <style> div[class^='block-container'] { padding-top: 2rem; } </style> ", True) # main area
            elif wch_section == 'sidebar': st.markdown(" <style> div[class^='st-emotion-cache-10oheav'] { padding-top: 0rem; } </style> ", True) # sidebar
            elif wch_section == 'all': 
                st.markdown(" <style> div[class^='block-container'] { padding-top: 2rem; } </style> ", True) # main area
                st.markdown(" <style> div[class^='st-emotion-cache-10oheav'] { padding-top: 0rem; } </style> ", True) # sidebar
            
        

            

        
            
            

                

            

            

        def ReadPictureFile(wch_fl):
            try:
                pxfl = f"{vpth}{wch_fl}"
                return base64.b64encode(open(pxfl, 'rb').read()).decode()

            except: return ""

        def PressedCheck(vcell):
            if mystate.plyrbtns[vcell]['isPressed'] == False:
                mystate.plyrbtns[vcell]['isPressed'] = True
                mystate.expired_cells.append(vcell)

                if mystate.plyrbtns[vcell]['eMoji'] == mystate.sidebar_emoji:
                    mystate.plyrbtns[vcell]['isTrueFalse'] = True
                    mystate.myscore += 5

                    if mystate.GameDetails[0] == 'Easy': mystate.myscore += 5
                    elif mystate.GameDetails[0] == 'Medium': mystate.myscore += 3
                    elif mystate.GameDetails[0] == 'Hard': mystate.myscore += 1
                
                else:
                    mystate.plyrbtns[vcell]['isTrueFalse'] = False
                    mystate.myscore -= 1

        def ResetBoard():
            total_cells_per_row_or_col = mystate.GameDetails[2]

            sidebar_emoji_no = random.randint(1, len(mystate.emoji_bank))-1
            mystate.sidebar_emoji = mystate.emoji_bank[sidebar_emoji_no]

            sidebar_emoji_in_list = False
            for vcell in range(1, ((total_cells_per_row_or_col ** 2)+1)):
                rndm_no = random.randint(1, len(mystate.emoji_bank))-1
                if mystate.plyrbtns[vcell]['isPressed'] == False:
                    vemoji = mystate.emoji_bank[rndm_no]
                    mystate.plyrbtns[vcell]['eMoji'] = vemoji
                    if vemoji == mystate.sidebar_emoji: sidebar_emoji_in_list = True

            if sidebar_emoji_in_list == False:  # sidebar pix is not on any button; add pix randomly
                tlst = [x for x in range(1, ((total_cells_per_row_or_col ** 2)+1))]
                flst = [x for x in tlst if x not in mystate.expired_cells]
                if len(flst) > 0:
                    lptr = random.randint(0, (len(flst)-1))
                    lptr = flst[lptr]
                    mystate.plyrbtns[lptr]['eMoji'] = mystate.sidebar_emoji

        def PreNewGame():
            total_cells_per_row_or_col = mystate.GameDetails[2]
            mystate.expired_cells = []
            mystate.myscore = 0

            
        
            foods = ['🍏', '🍎', '🍐', '🍊', '🍋', '🍌', '🍉', '🍇', '🍓', '🍈', '🍒', '🍑', '🥭', '🍍', '🥥', '🥝', '🍅', '🍆', '🥑', '🥦', '🥬', '🥒', '🌽', '🥕', '🧄', '🧅', '🥔', '🍠', '🥐', '🥯', '🍞', '🥖', '🥨', '🧀', '🥚', '🍳', '🧈', '🥞', '🧇', '🥓', '🥩', '🍗', '🍖', '🦴', '🌭', '🍔', '🍟', '🍕']
            clocks = ['🕓', '🕒', '🕑', '🕘', '🕛', '🕚', '🕖', '🕙', '🕔', '🕤', '🕠', '🕕', '🕣', '🕞', '🕟', '🕜', '🕢', '🕦']
            hands = ['🤚', '🖐', '✋', '🖖', '👌', '🤏', '✌️', '🤞', '🤟', '🤘', '🤙', '👈', '👉', '👆', '🖕', '👇', '☝️', '👍', '👎', '✊', '👊', '🤛', '🤜', '👏', '🙌', '🤲', '🤝', '🤚🏻', '🖐🏻', '✋🏻', '🖖🏻', '👌🏻', '🤏🏻', '✌🏻', '🤞🏻', '🤟🏻', '🤘🏻', '🤙🏻', '👈🏻', '👉🏻', '👆🏻', '🖕🏻', '👇🏻', '☝🏻', '👍🏻', '👎🏻', '✊🏻', '👊🏻', '🤛🏻', '🤜🏻', '👏🏻', '🙌🏻', '🤚🏽', '🖐🏽', '✋🏽', '🖖🏽', '👌🏽', '🤏🏽', '✌🏽', '🤞🏽', '🤟🏽', '🤘🏽', '🤙🏽', '👈🏽', '👉🏽', '👆🏽', '🖕🏽', '👇🏽', '☝🏽', '👍🏽', '👎🏽', '✊🏽', '👊🏽', '🤛🏽', '🤜🏽', '👏🏽', '🙌🏽']
            animals = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨', '🐯', '🦁', '🐮', '🐷', '🐽', '🐸', '🐵', '🙈', '🙉', '🙊', '🐒', '🐔', '🐧', '🐦', '🐤', '🐣', '🐥', '🦆', '🦅', '🦉', '🦇', '🐺', '🐗', '🐴', '🦄', '🐝', '🐛', '🦋', '🐌', '🐞', '🐜', '🦟', '🦗', '🦂', '🐢', '🐍', '🦎', '🦖', '🦕', '🐙', '🦑', '🦐', '🦞', '🦀', '🐡', '🐠', '🐟', '🐬', '🐳', '🐋', '🦈', '🐊', '🐅', '🐆', '🦓', '🦍', '🦧', '🐘', '🦛', '🦏', '🐪', '🐫', '🦒', '🦘', '🐃', '🐂', '🐄', '🐎', '🐖', '🐏', '🐑', '🦙', '🐐', '🦌', '🐕', '🐩', '🦮', '🐕‍🦺', '🐈', '🐓', '🦃', '🦚', '🦜', '🦢', '🦩', '🐇', '🦝', '🦨', '🦦', '🦥', '🐁', '🐀', '🦔']
            vehicles = ['🚗', '🚕', '🚙', '🚌', '🚎', '🚓', '🚑', '🚒', '🚐', '🚚', '🚛', '🚜', '🦯', '🦽', '🦼', '🛴', '🚲', '🛵', '🛺', '🚔', '🚍', '🚘', '🚖', '🚡', '🚠', '🚟', '🚃', '🚋', '🚞', '🚝', '🚄', '🚅', '🚈', '🚂', '🚆', '🚇', '🚊', '🚉', '✈️', '🛫', '🛬', '💺', '🚀', '🛸', '🚁', '🛶', '⛵️', '🚤', '🛳', '⛴', '🚢']
            
            moon = ['🌕', '🌔', '🌓', '🌗', '🌒', '🌖', '🌑', '🌜', '🌛', '🌙']

            random.seed()
            if mystate.GameDetails[0] == 'Easy':
                wch_bank = random.choice(['foods'])
                mystate.emoji_bank = locals()[wch_bank]

            
            mystate.plyrbtns = {}
            for vcell in range(1, ((total_cells_per_row_or_col ** 2)+1)): mystate.plyrbtns[vcell] = {'isPressed': False, 'isTrueFalse': False, 'eMoji': ''}

        def ScoreEmoji():
            if mystate.myscore == 0: return '😐'
            elif -5 <= mystate.myscore <= -1: return '😏'
            elif -10 <= mystate.myscore <= -6: return '☹️'
            elif mystate.myscore <= -11: return '😖'
            elif 1 <= mystate.myscore <= 5: return '🙂'
            elif 6 <= mystate.myscore <= 10: return '😊'
            elif mystate.myscore > 10: return '😁'

        def NewGame():
            ResetBoard()
            total_cells_per_row_or_col = mystate.GameDetails[2]

            ReduceGapFromPageTop('sidebar')
            
            

            st.markdown(sbe.replace('|fill_variable|', mystate.sidebar_emoji), True)

            aftimer = st_autorefresh(interval=(mystate.GameDetails[1] * 1000), key="aftmr")
            if aftimer > 0: mystate.myscore -= 1

            st.info(f"{ScoreEmoji()} Score: {mystate.myscore} | Pending: {(total_cells_per_row_or_col ** 2)-len(mystate.expired_cells)}")

            st.markdown(horizontal_bar, True)
            
            
            

            # Set Board Dafaults
            st.markdown("<style> div[class^='css-1vbkxwb'] > p { font-size: 1.5rem; } </style> ", unsafe_allow_html=True)  # make button face big

            for i in range(1, (total_cells_per_row_or_col+1)):
                tlst = ([1] * total_cells_per_row_or_col) + [2] # 2 = rt side padding
                globals()['cols' + str(i)] = st.columns(tlst)
            
            for vcell in range(1, (total_cells_per_row_or_col ** 2)+1):
                if 1 <= vcell <= (total_cells_per_row_or_col * 1):
                    arr_ref = '1'
                    mval = 0

                elif ((total_cells_per_row_or_col * 1)+1) <= vcell <= (total_cells_per_row_or_col * 2):
                    arr_ref = '2'
                    mval = (total_cells_per_row_or_col * 1)

                elif ((total_cells_per_row_or_col * 2)+1) <= vcell <= (total_cells_per_row_or_col * 3):
                    arr_ref = '3'
                    mval = (total_cells_per_row_or_col * 2)

                elif ((total_cells_per_row_or_col * 3)+1) <= vcell <= (total_cells_per_row_or_col * 4):
                    arr_ref = '4'
                    mval = (total_cells_per_row_or_col * 3)

                elif ((total_cells_per_row_or_col * 4)+1) <= vcell <= (total_cells_per_row_or_col * 5):
                    arr_ref = '5'
                    mval = (total_cells_per_row_or_col * 4)

                elif ((total_cells_per_row_or_col * 5)+1) <= vcell <= (total_cells_per_row_or_col * 6):
                    arr_ref = '6'
                    mval = (total_cells_per_row_or_col * 5)

                elif ((total_cells_per_row_or_col * 6)+1) <= vcell <= (total_cells_per_row_or_col * 7):
                    arr_ref = '7'
                    mval = (total_cells_per_row_or_col * 6)

                elif ((total_cells_per_row_or_col * 7)+1) <= vcell <= (total_cells_per_row_or_col * 8):
                    arr_ref = '8'
                    mval = (total_cells_per_row_or_col * 7)

                elif ((total_cells_per_row_or_col * 8)+1) <= vcell <= (total_cells_per_row_or_col * 9):
                    arr_ref = '9'
                    mval = (total_cells_per_row_or_col * 8)

                elif ((total_cells_per_row_or_col * 9)+1) <= vcell <= (total_cells_per_row_or_col * 10):
                    arr_ref = '10'
                    mval = (total_cells_per_row_or_col * 9)
                    
                globals()['cols' + arr_ref][vcell-mval] = globals()['cols' + arr_ref][vcell-mval].empty()
                if mystate.plyrbtns[vcell]['isPressed'] == True:
                    if mystate.plyrbtns[vcell]['isTrueFalse'] == True:
                        globals()['cols' + arr_ref][vcell-mval].markdown(pressed_emoji.replace('|fill_variable|', '✅️'), True)
                    
                    elif mystate.plyrbtns[vcell]['isTrueFalse'] == False:
                        globals()['cols' + arr_ref][vcell-mval].markdown(pressed_emoji.replace('|fill_variable|', '❌'), True)

                else:
                    vemoji = mystate.plyrbtns[vcell]['eMoji']
                    globals()['cols' + arr_ref][vcell-mval].button(vemoji, on_click=PressedCheck, args=(vcell, ), key=f"B{vcell}")

            st.caption('') # vertical filler
            st.markdown(horizontal_bar, True)

            if len(mystate.expired_cells) == (total_cells_per_row_or_col ** 2):
                

                if mystate.myscore > 0: st.balloons()
                elif mystate.myscore <= 0: st.snow()

                tm.sleep(5)
                mystate.runpage = Main
                st.rerun()

        def Main():
            st.markdown('<style>[data-testid="stSidebar"] > div:first-child {width: 310px;}</style>', unsafe_allow_html=True,)  # reduce sidebar width
            st.markdown(purple_btn_colour, unsafe_allow_html=True)

            
            
        mystate.GameDetails[0] = 'Easy'

        st.subheader(' PICMATCHING')
        st.write(':orange[PicMatch - Match emojis scattered across the grid to the ones displayed in the image]')

        _LOREM_IPSUM = """Engage in a dynamic blend of memory exercises, problem-solving tasks, and attention training in this immersive and challenging game."""
        def stream_data():
            for word in _LOREM_IPSUM.split(" "):
                    yield word + " "
                    time.sleep(0.08)
        if st.form_submit_button("Why play this game"):
                st.write_stream(stream_data)
        st.write('')
                

        if st.form_submit_button(f"🕹️ Start the  Game", use_container_width=True):

            if mystate.GameDetails[0] == 'Easy':
                mystate.GameDetails[1] = 20         # secs interval
                mystate.GameDetails[2] = 6         # total_cells_per_row_or_col
                    
                    
                    

                PreNewGame()
                mystate.runpage = NewGame
                st.rerun()

        st.markdown(horizontal_bar, True)


    if 'runpage' not in mystate: mystate.runpage = Main
    mystate.runpage()

if __name__ == "__main__":
    app()
