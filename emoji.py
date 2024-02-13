import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
emoji_dist = {0: "data/emojis/angry.png", 1: "data/emojis/disgust.png", 2: "data/emojis/fear.png", 3: "data/emojis/happy.png", 4: "data/emojis/neutral.png", 5: "data/emojis/sad.png", 6: "data/emojis/surprised.png"}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]
global frame_number

cap1 = cv2.VideoCapture(0)

def show_subject(frame_num):
    global cap1
    if not cap1.isOpened():
        print("cant open the camera1")
        cap1.release()
        exit()

    cap1.set(1, frame_num)
    flag1, frame1 = cap1.read()

    if frame1 is None or frame1.size == 0:
        print("Empty frame1")
        return

    frame1 = cv2.resize(frame1, (600, 500))
    bounding_box = cv2.CascadeClassifier(
        '/Users/prakhyath/Desktop/ML Project/.venv/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)

        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        show_text[0] = maxindex

    global last_frame1
    last_frame1 = frame1.copy()
    pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10,  show_subject(frame_num))

def show_avatar():
    emoji_path = emoji_dist.get(show_text[0], "data/emojis/*")
    frame2 = cv2.imread(emoji_path)

    # Ensure the emoji image has the correct color representation
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    pic2 = Image.fromarray(frame2)
    imgtk2 = ImageTk.PhotoImage(image=pic2)

    lmain2.imgtk2 = imgtk2
    lmain2.configure(image=imgtk2)
    root.update()

    lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))

    root.after(10, show_avatar)

    


if __name__ == '__main__':
    frame_number = 0
    root = tk.Tk()

    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)


    
threading.Thread(target=show_subject, args=(frame_number,)).start()
threading.Thread(target=show_avatar).start()
root.mainloop()
