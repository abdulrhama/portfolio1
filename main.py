import cv2
import numpy as np
import time
from datetime import datetime
import winsound
import cvzone
from typing import Any
from numpy import ndarray
def stackImages(scale,imgArray):
    rows =len(imgArray)
    cols=len(imgArray[0])
    rowsAvailable =isinstance(imgArray[0],list)
    Width= imgArray[0][0].shape[1]
    Height =imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range( 0,rows):
            for y in range (0,cols):
                if imgArray[x][y].shape[:2]== imgArray[0][0].shape[:2]:
                    imgArray[x][y]=cv2.resize(imgArray[x][y],(0,0),None,scale,scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y],(imgArray[0][0].shape[1],imgArray[0][0].shape[0]),None,scale,scale)
                if len(imgArray[x][y].shape)==2:imgArray[x][y]= cv2.cvtColor ( imgArray[x][y],cv2.COLOR_GRAY2BGR)
        imageBlank=np.zeros((Height,Width,3),np.uint8)
        hor=[imageBlank]*rows
        hor_con=[imageBlank]*rows
        for x in range(0,rows):
            hor[x]=np.hstack(imgArray[x])
        ver=np.vstack(hor)
    else:
        for x in range(0,rows):
            if imgArray[x].shape[2:]==imgArray[0].shape[:2]:
                imgArray[x]=cv2.resize(imgArray[x],(0,0),None,scale,scale)
            else:
                imgArray[x]=cv2.resize(imgArray[x],(imgArray[0].shape[1],imgArray[0].shape[0]),None,scale,scale)
            if len(imgArray[x].shape)==2:imgArray[x]=cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)
        hor=np.hstack(imgArray)
        ver =hor
    return ver
def show_login_window():
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    main_program()

def main_program():
    print("Welcome to the main program!")

if __name__ == "__main__":
    show_login_window()
def start_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        exit()
fgbg = cv2.createBackgroundSubtractorMOG2()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
motion_threshold = 500
wait_time = 0

previous_frame = None
motion_threshold = 5000
wait_time = 0.1
is_recording = False
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
cap = cv2.VideoCapture(0)
previous_frame = None
if not cap.isOpened():
    print("Could not open camera.")
    exit()
ret, frame1 = cap.read()
ret, frame2 = cap.read()
ret, frame3 = cap.read()
while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_frame is None:
        previous_frame = gray_frame
        continue
    frame_diff = cv2.absdiff(previous_frame, gray_frame)
    _, threshold_frame = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    dilated_frame = cv2.dilate(threshold_frame, None, iterations=2)
    contours, _ = cv2.findContours(dilated_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    if total_area > motion_threshold:
        if not is_recording:
            is_recording = True
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            out = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    else:
        if is_recording:
            is_recording = False
            out.release()
            print(f"تم حفظ الفيديو: {filename}")
    if is_recording:
        out.write(frame)
    diff=cv2.absdiff(frame1,frame2)
    gray=cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _,thresh =cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated =cv2.dilate(thresh,None,iterations=3)
    contours,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c)<7000:
            continue
        x,y,w,h=cv2.boundingRect(c)
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
        winsound.PlaySound('sound5.mp3',winsound.SND_ASYNC)
    if cv2.waitKey(10)==ord('q'):
        break
    ret, frame4 = cap.read()
    gray = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame4, (x, y), (x+w, y+h), (255, 0, 0), 2)
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    blur = cv2.GaussianBlur(fgmask, (5, 5), 0)
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, black_and_white = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    frame1 = frame2
    ret, frame2 = cap.read()
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_frame is None:
        previous_frame = gray_frame
        continue
    frame_delta = cv2.absdiff(previous_frame, gray_frame)
    _, threshold_delta = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_delta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > motion_threshold:
            motion_detected = True
            break
    if motion_detected:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Motion detected at {current_time}")
        time.sleep(wait_time)
    previous_frame = gray_frame
    stackedImages=stackImages(0.5 ,([frame,frame1,frame4,black_and_white,blur],

                               ))
    cv2.imshow("Result",stackedImages)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
    # cv2.imshow('Face Tracking', frame4)
    # cv2.imshow("Motion Detection (Black and White)", black_and_white)
    # cv2.imshow('Motion Detection', blur)
    # cv2.imshow('granny cam',frame1)
    # cv2.imshow('Motion Detection', frame)
    # imgList =[frame,frame1,black_and_white,blur,frame4]
    # stackedImg =cvzone.stackImages(imgList, cols=2, scale=0.5)
    # cv2.imshow("stackedImg",stackedImg)
