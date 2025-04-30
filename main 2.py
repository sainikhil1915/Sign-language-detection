import math
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import tkinter as tk

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

current_letter = ""

# Function to update the textbox with the recognized letter
def store_letter(event):
    if event.char.lower() == 'z':
        current_text = text_box.get("1.0", tk.END).strip()
        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, current_text + current_letter)

# Function to exit the application
def exit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Create the Tkinter window
root = tk.Tk()
root.title("Hand Gesture Recognition")

# Create the textbox
text_box = tk.Text(root, height=2, width=30, font=("Helvetica", 16))
text_box.pack(pady=10)

# Bind the 'z' key to store the letter
root.bind("<Key>", store_letter)

# Create the Exit button
exit_button = tk.Button(root, text="Exit", command=exit_app, font=("Helvetica", 16), bg="lightcoral")
exit_button.pack(pady=10)

# Function to process the video feed and update the GUI
def process_video():
    global current_letter
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        current_letter = labels[index]
        cv2.putText(imgOutput, current_letter, (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow('Image', imgOutput)
    root.after(10, process_video)

# Start processing the video
process_video()

# Run the Tkinter main loop
root.mainloop()
