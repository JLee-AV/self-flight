
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configargparse

import cv2
import numpy as np
from time import sleep

from Gestures.tello_gesture_controller import TelloGestureController
from Gestures.gesture_recognition import GestureRecognition
from Gestures.gesture_recognition import GestureBuffer

from utils.cvfpscalc import CvFpsCalc

from djitellopy import Tello
from Gestures import *

import threading


hsvValshoop = [70,101,99,179,255,255]  #[0,100,0,40,255,255] for red    #[40,100,100,85,255,255] for green

hsvValswall = [0,108,67,179,255,255]

hsvValsg = [35,72,0,77,255,255]
hsvValsr = [0,100,0,33,255,255]

sensors = 3

threshold = 0.2

width, height = 480, 360

senstivity = 3  # if number is high less sensitive

weights = [-25, -15, 0, 15, 25]

fSpeed = 0

curve = 0

Island = 0

#global variable t is a mean to switch from hulahoop to handgesture.
t=0
k=0
s=0

tello = Tello()
tello.connect()
print(tello.get_battery())



def thresholdinglinefw(img,color):
    if color == 'green':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([hsvValsg[0], hsvValsg[1], hsvValsg[2]])

        upper = np.array([hsvValsg[3], hsvValsg[4], hsvValsg[5]])

        mask = cv2.inRange(hsv, lower, upper)

        return mask
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([hsvValsr[0], hsvValsr[1], hsvValsr[2]])

        upper = np.array([hsvValsr[3], hsvValsr[4], hsvValsr[5]])

        mask = cv2.inRange(hsv, lower, upper)

        return mask



def colordetect(img): #this function tell you which either green/red color of the line is. and with that info, hsvvals array will be assigned.

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([hsvValsg[0], hsvValsg[1], hsvValsg[2]])

    upper = np.array([hsvValsg[3], hsvValsg[4], hsvValsg[5]])

    mask = cv2.inRange(hsv, lower, upper)

    cx = 0

    contours, hieracrhy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(biggest)

        if w > 10 and h > 10:
            return 'green'

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([hsvValsr[0], hsvValsr[1], hsvValsr[2]])

    upper = np.array([hsvValsr[3], hsvValsr[4], hsvValsr[5]])

    mask = cv2.inRange(hsv, lower, upper)

    cx = 0

    contours, hieracrhy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(biggest)

        if w > 10 and h > 10:
            return 'red'


def getContoursflightline1():
  n = 0
  while True:

    img = tello.get_frame_read().frame

    img = cv2.resize(img, (width, height))

    img = cv2.flip(img, 0)

    if n==0: #it determines color at the  beginning and keep looking only at the color
        color = colordetect(img)
        print(color)
        n=n+1

    imgThres = thresholdinglinefw(img, color)

    cv2.imshow("Output", img)

    cv2.imshow("Path", imgThres)

    cv2.waitKey(1)

    cx = 0

    contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(biggest)

        cx = x + w // 2

        cy = y + h // 2

        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)

        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    if len(contours) != 0:

        if cx >= 249:
            tello.send_rc_control(16, 0, 0, 0)
        if cx <= 231:
            tello.send_rc_control(-7, 0, 0, 0)
        if 231 < cx < 249:
            tello.send_rc_control(0, 27, 0, 0)
            sleep(0.3)
        if w > 210:
            tello.move_forward(20)
            tello.rotate_counter_clockwise(90)
            tello.send_rc_control(0, 26, 0, 0)
            sleep(4.3)
            tello.send_rc_control(0, 0, 0, 0)
            break


def getContoursflightlinediff():
    n = 0
    while True:

        img = tello.get_frame_read().frame

        img = cv2.resize(img, (width, height))

        img = cv2.flip(img, 0)

        if n == 0:  # it determines color at the  beginning and keep looking only at the color
            color = colordetect(img)
            print(color)
            n = n + 1

        imgThres = thresholdinglinefw(img, color)

        cv2.imshow("Output", img)

        cv2.imshow("Path", imgThres)

        cv2.waitKey(1)

        cx = 0

        contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            biggest = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(biggest)

            cx = x + w // 2

            cy = y + h // 2

            cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)

            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # monitoring and align the heading onto line(center circle)
        if len(contours) != 0:

            if cx >= 247:
                tello.send_rc_control(18, 0, 0, 0)
            if cx <= 233:
                tello.send_rc_control(-10, 0, 0, 0)
            if 233 < cx < 247:
                tello.send_rc_control(0, 30, 0, 0)
                sleep(0.2)
            if y > 130:
                tello.send_rc_control(0, 0, 0, 0)
                tello.rotate_counter_clockwise(90)
                tello.move_up(74)
                tello.send_rc_control(0,25,0,0)
                sleep(3)
                break


def getContoursflightline3():
    n = 0
    while True:

        img = tello.get_frame_read().frame

        img = cv2.resize(img, (width, height))

        img = cv2.flip(img, 0)

        if n == 0:  # it determines color at the  beginning and keep looking only at the color
            color = colordetect(img)
            print(color)
            n = n + 1

        imgThres = thresholdinglinefw(img, color)

        cv2.imshow("Output", img)

        cv2.imshow("Path", imgThres)

        cv2.waitKey(1)

        cx = 0

        contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            biggest = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(biggest)

            cx = x + w // 2

            cy = y + h // 2

            cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)

            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        if len(contours) != 0:

            if cx >= 243:
                tello.send_rc_control(12, 5, 0, 0)
            if cx <= 237:
                tello.send_rc_control(-10, 5, 0, 0)
            if 237 < cx < 243:
                tello.send_rc_control(0, 30, 0, 0)
                sleep(0.3)
            if w > 200:
                tello.move_forward(47)
                tello.rotate_clockwise(90)
                tello.send_rc_control(0, 30, 0, 0)
                sleep(2.7)
                tello.send_rc_control(0, 0, 0, 0)
                break


def getContoursflightlinelast():
    n = 0
    while True:

        img = tello.get_frame_read().frame

        img = cv2.resize(img, (width, height))

        img = cv2.flip(img, 0)

        if n == 0:  # it determines color at the  beginning and keep looking only at the color
            color = colordetect(img)
            print(color)
            n = n + 1

        imgThres = thresholdinglinefw(img, color)

        cv2.imshow("Output", img)

        cv2.imshow("Path", imgThres)

        cv2.waitKey(1)

        cx = 0

        contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            biggest = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(biggest)

            cx = x + w // 2

            cy = y + h // 2

            cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)

            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # monitoring and align the heading onto line(center circle)
        if len(contours) != 0:

            #droping mirror mechanism here

            if w<200:
                if cx >= 249:
                    tello.send_rc_control(11, 5, 0, 0)
                if cx <= 231:
                    tello.send_rc_control(-8, 5, 0, 0)
                if 231 < cx < 249:
                    tello.send_rc_control(0, 30, 0, 0)
                    sleep(0.2)
                if y > 180 or h<30:
                    tello.send_rc_control(0, 0, 0, 0)
                    tello.move_up(60)
                    sleep(1)
                    tello.move_down(170)
                    tello.send_rc_control(0,-100,0,0)
                    sleep(5)
                    tello.send_rc_control(0, 0, 0, 0)
                    tello.move_up(110)
                    break
            if w>200:
                tello.send_rc_control(0,19,0,0)

def wallthresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([hsvValswall[0], hsvValswall[1], hsvValswall[2]])

    upper = np.array([hsvValswall[3], hsvValswall[4], hsvValswall[5]])

    mask = cv2.inRange(hsv, lower, upper)

    return mask


def wallgetContoursflight(imgThres, img):
    cx = 0
    cy = 0

    contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:

        biggest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(biggest)

        cx = x + w // 2

        cy = y + h // 2

        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)

        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    if len(contours) != 0:
        tello.send_rc_control(0, 20, 0, 0)

        if y < 15:
            cv2.destroyAllWindows()
            while True:
                img = tello.get_frame_read().frame

                img = cv2.resize(img, (width, height))

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                lower = np.array([hsvValshoop[0], hsvValshoop[1], hsvValshoop[2]])

                upper = np.array([hsvValshoop[3], hsvValshoop[4], hsvValshoop[5]])

                mask = cv2.inRange(hsv, lower, upper)

                cx = 0

                contours, hieracrhy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                tello.send_rc_control(-20, 0, 0, 0)

                if len(contours) != 0:
                    biggest = max(contours, key=cv2.contourArea)

                    x, y, w, h = cv2.boundingRect(biggest)

                    if w > 40 and h > 30:
                        tello.send_rc_control(0,0,0,0)
                        global k
                        k=1
                        cv2.destroyAllWindows()
                        break



def get_args():
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])

    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add("--device", type=int)
    parser.add("--width", help='cap width', type=int)
    parser.add("--height", help='cap height', type=int)
    parser.add("--is_keyboard", help='To use Keyboard control by default', type=bool)
    parser.add('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add("--min_detection_confidence",
               help='min_detection_confidence',
               type=float)
    parser.add("--min_tracking_confidence",
               help='min_tracking_confidence',
               type=float)
    parser.add("--buffer_len",
               help='Length of gesture buffer',
               type=int)

    args = parser.parse_args()

    return args


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([hsvValshoop[0], hsvValshoop[1], hsvValshoop[2]])

    upper = np.array([hsvValshoop[3], hsvValshoop[4], hsvValshoop[5]])

    mask = cv2.inRange(hsv, lower, upper)

    return mask


def getContoursflight(imgThres, img):
    cx = 0
    cy = 0

    contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(biggest)

        cx = x + w // 2

        cy = y + h // 2

        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)

        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    #below : flight control section
    if len(contours) != 0:
        if w > 300 or h > 300:
            tello.send_rc_control(0, 35, -25, 0)
            sleep(2.5)
            tello.send_rc_control(0, 0, 0, 0)
            global t
            t = 1

        if cx > 240 and cy > 180:                    # rc control parameter : LR/BackFfwrd/DwnUp/Yaw
           tello.send_rc_control(20, 15, -20, curve)  # control parameter + : right,forward,up,yaw2right.      - : left,backward,down,yaw2left
        elif cx > 240 and cy < 180:
           tello.send_rc_control(20, 15, 20, curve)
        elif cx < 240 and cy > 180:
           tello.send_rc_control(-20, 15, -20, curve)
        elif cx < 240 and cy < 180:
           tello.send_rc_control(-20, 15, 20, curve)
        else:
           tello.send_rc_control(0, 0, 0, 0)



def getContoursflight2(imgThres, img):
    cx = 0
    cy = 0

    contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(biggest)

        cx = x + w // 2

        cy = y + h // 2

        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)

        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    #below : flight control section
    if len(contours) != 0:
        if w > 300 or h > 300:
            tello.send_rc_control(0, 35, -25, 0)
            sleep(2.5)
            tello.send_rc_control(0, 0, 0, 0)
            global s
            s = 1

        if cx > 240 and cy > 180:                    # rc control parameter : LR/BackFfwrd/DwnUp/Yaw
           tello.send_rc_control(20, 15, -20, curve)  # control parameter + : right,forward,up,yaw2right.      - : left,backward,down,yaw2left
        elif cx > 240 and cy < 180:
           tello.send_rc_control(20, 15, 20, curve)
        elif cx < 240 and cy > 180:
           tello.send_rc_control(-20, 15, -20, curve)
        elif cx < 240 and cy < 180:
           tello.send_rc_control(-20, 15, 20, curve)
        else:
           tello.send_rc_control(0, 0, 0, 0)



def hulahoop():
    img = tello.get_frame_read().frame

    img = cv2.resize(img, (width, height))

    imgThres = thresholding(img)

    getContoursflight(imgThres, img)

    if t==1:
        cv2.destroyAllWindows()

    if t == 0:
        if t==1:
            cv2.destroyAllWindows()
        cv2.imshow("Output", img)

        cv2.imshow("Path", imgThres)

        cv2.waitKey(1)


def hulahoop2():
    img = tello.get_frame_read().frame

    img = cv2.resize(img, (width, height))

    imgThres = thresholding(img)

    getContoursflight2(imgThres, img)

    if s==1:
        cv2.destroyAllWindows()

    if s == 0:
        if s==1:
            cv2.destroyAllWindows()
        cv2.imshow("Output", img)

        cv2.imshow("Path", imgThres)

        cv2.waitKey(1)


def main():


    tello.streamon()

    tello.takeoff()

    cap = cv2.VideoCapture(1)

    getContoursflightline1()
    sleep(1)
    getContoursflightlinediff()
    sleep(1)
    getContoursflightline3()
    sleep(1)
    getContoursflightlinelast()

    sleep(2)

    while True:
        img = tello.get_frame_read().frame

        img = cv2.resize(img, (width, height))

        imgThres = wallthresholding(img)

        wallgetContoursflight(imgThres, img)

        cv2.imshow("Output", img)

        cv2.imshow("Path", imgThres)

        cv2.waitKey(1)

        if k==1:
            break

    sleep(1)

    while True:
       hulahoop()
       if t==1 :
           break

    while True:
        hulahoop2()
        if s==1 :
            break
    #transition hulahoop to handgest

    # init global vars
    global gesture_buffer
    global gesture_id
    global battery_status

    # Argument parsing
    args = get_args()
    KEYBOARD_CONTROL = args.is_keyboard
    WRITE_CONTROL = False
    in_flight = False

    # Init Tello Controllers
    gesture_controller = TelloGestureController(tello)

    gesture_detector = GestureRecognition(args.use_static_image_mode, args.min_detection_confidence,
                                          args.min_tracking_confidence)
    gesture_buffer = GestureBuffer(buffer_len=args.buffer_len)

    def tello_control(key, gesture_controller):
        global gesture_buffer
        gesture_controller.gesture_control(gesture_buffer)

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    mode = 0
    number = -1
    while True:
        fps = cv_fps_calc.get()

        # Process Key (ESC: end)
        key = cv2.waitKey(1) & 0xff
        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            if not in_flight:
                # Take-off drone
                tello.takeoff()
                in_flight = True

            elif in_flight:
                # Land tello
                tello.land()
                in_flight = False

        if WRITE_CONTROL:
            number = -1
            if 48 <= key <= 57:  # 0 ~ 9
                number = key - 48

        # Camera capture
        cap2 = tello.get_frame_read()
        image = cap2.frame

        debug_image, gesture_id = gesture_detector.recognize(image, number, mode)
        gesture_buffer.add_gesture(gesture_id)

        # Start control thread
        threading.Thread(target=tello_control, args=(key, gesture_controller,)).start()

        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)

        cv2.imshow('Tello Gesture Recognition', debug_image)

    # tello.land()#land by pressing esc
    tello.end()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
