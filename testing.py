import random

import cv2
import ev3_dc as ev3
from time import sleep
from datetime import datetime
import dice_detection
import DB_test

def GetImage(cam, reader, x, y, w, h, res_w, res_h):
    ret, frame = cam.read()
    center = dice_detection.get_frame_center(frame, x, y, w, h)

    cv2.rectangle(frame, (x, y), (x + w,y + h), (255, 255, 0), 2)

    toss = {}
    # blob detection
    dx, dy, dw, dh = dice_detection.detect_circle_coordinates(center, res_w, res_h)
    if dx == -1:

        now = datetime.now()
        time = now.strftime("%m%d%Y_%H%M%S")

        toss = {
            'toss': 'X',
            'time': time,
            'filename': '-'
        }
        return toss
    dice = center[dy:dy + dh, dx:dx + dw]
    cv2.rectangle(frame, (dx + x, dy + y), (dx + x + dw, dy + y + dh), (255, 0, 0), 2)
    cv2.putText(frame, 'DICE', (dx + x, dy + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    #
    # dim = (256, 256)
    # # resize image
    # resized_dice = cv2.resize(dice, dim, interpolation=cv2.INTER_AREA)
    #
    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # # resize image
    smaller_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Dice detection", smaller_frame)
    #
    # detected_letter = dice_detection.test_out_rotations(resized_dice, reader)
    # cv2.putText(frame, detected_letter, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #
    # now = datetime.now()
    # time = now.strftime("%m%d%Y_%H%M%S")
    #
    # #letter not detected
    # if detected_letter == None:
    #     detected_letter = 'X'
    #
    # toss = {
    #         'toss': detected_letter,
    #         'time': time,
    #         'data': resized_dice,
    #         'filename': detected_letter + "_" + time + "_" + str(random.randint(0,255)) + '.png'
    #     }

    return toss

def main():

    # set variables for openCV and OCR
    cam = cv2.VideoCapture(0)
    reader = dice_detection.initialize_ocr_reader()
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    res_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    res_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    x = int((650/1920) * res_w)
    y = int((250/1080) * res_h)
    w = int((750/1920) * res_w)
    h = int((750/1080) * res_h)

    while True:

        toss = GetImage(cam, reader, x, y, w, h, res_w, res_h)

        # end with whatever key pressed
        if cv2.waitKey(1) != -1:
            break

        pass

    # release all OpenCV thingies
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
