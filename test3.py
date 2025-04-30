import math
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Initialize the hand detector and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

detected_text = ""  # Variable to hold detected text

# Initialize Tkinter window
root = tk.Tk()
root.title("Real Time Sign Detection")

# Create a label to display the detected text
detected_label = tk.Label(root, text="Text Box:", font=("Helvetica", 16))
detected_label.pack()

detected_text_box = tk.Text(root, height=2, font=("Helvetica", 16))
detected_text_box.pack()

# Function to update the Tkinter window with the video feed
def update_frame():
    global detected_text
    success, img = cap.read()
    if not success:
        return

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
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # Convert the image to a format Tkinter can use
    imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgRGB)
    imgTk = ImageTk.PhotoImage(imgPIL)

    # Update the label with the new image
    if not hasattr(update_frame, 'label_img'):
        update_frame.label_img = tk.Label(root)
        update_frame.label_img.pack()
    update_frame.label_img.config(image=imgTk)
    update_frame.label_img.image = imgTk

    # Update the Tkinter window
    root.update_idletasks()
    root.after(10, update_frame)

# Function to handle key press events
def on_key_press(event):
    global detected_text
    if event.char == 'z':
        success, img = cap.read()
        if success:
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
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                detected_text += labels[index]
                detected_text_box.insert(tk.END, labels[index])
                detected_text_box.see(tk.END)  # Scroll to the end

# Function to exit the application
def exit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Bind the key press event to the function
root.bind('<KeyPress>', on_key_press)

# Create the Exit button
exit_button = tk.Button(root, text="Exit", command=exit_app, font=("Helvetica", 16), bg="lightcoral")
exit_button.pack(pady=10)

# Start the video feed update loop
update_frame()

# Start the Tkinter event loop
root.mainloop()
