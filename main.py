import os
import random

import cv2
import ev3_dc as ev3
from time import sleep
from datetime import datetime
import dice_detection
import DB_test
import camera
import threading

def DiceItRirect(my_ev3):

    my_ev3.movement_plan = (
                my_ev3.move_to(-30, speed=100, ramp_up=100, ramp_down=100, brake=True) +
                my_ev3.move_to(0, speed=100, ramp_up=100, ramp_down=100, brake=True) +
                my_ev3.stop_as_task(brake=False)
        )

    my_ev3.movement_plan.start(thread=False)
    pass

def GetImage(cam, reader, x, y, w, h, res_w, res_h, model):
    frame = cam.read()

    center = dice_detection.get_frame_center(frame, x, y, w, h)

    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
    cv2.imshow("Dice location", frame)

    # blob detection
    dx, dy, dw, dh = dice_detection.detect_circle_coordinates(center, res_w, res_h)
    if dx == -1:

        now = datetime.now()
        time = now.strftime("%m%d%Y_%H%M%S")

        toss = {
            'toss': '-',
            'time': time,
            'filename': '-',
            'probability': '-'
        }
        return toss
    dice = center[dy:dy + dh, dx:dx + dw]

    dim = (256, 256)
    # resize image
    resized_dice = cv2.resize(dice, dim, interpolation=cv2.INTER_AREA)

    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    detected_letter_nn, probability = dice_detection.get_nn_label(model, resized_dice)

    now = datetime.now()
    time = now.strftime("%m%d%Y_%H%M%S")

    #letter not detected
    if detected_letter_nn == None or probability < 0.8:
        detected_letter_nn = 'X'

    toss = {
            'toss': detected_letter_nn,
            'time': time,
            'data': resized_dice,
            'filename': detected_letter_nn + "_" + time + "_" + str(random.randint(0,255)) + '.png',
            'probability': probability
        }

    return toss


# saves the image to specified path, which should have subdirectiories:
#A, B, C, D, E, F, X
def save_image(path, toss):

    if 'data' in toss.keys():
        filepath = os.path.join(path,toss['toss'],toss['filename'])
        cv2.imwrite(filepath, toss['data'])

def main():

    # output photos will be saved here
    # requires subdirectiories A, B, C, D, E, F, X
    dataset_output_path = 'out'

    # connector to EV3 brick
    my_ev3 = ev3.Motor(
        ev3.PORT_D,
        protocol=ev3.USB
    )
    #my_ev3.sync_mode = ev3.SYNC

    # db connection
    # parameters of the connection can be passed in connect arguments
    cnx = DB_test.connect()

    # set variables for openCV and OCR
    cam = camera.CameraCapture(0)
    reader = dice_detection.initialize_ocr_reader()
    res_w = cam.get_w()
    res_h = cam.get_h()

    x = int((650 / 1920) * res_w)
    y = int((250 / 1080) * res_h)
    w = int((750 / 1920) * res_w)
    h = int((750 / 1080) * res_h)

    model = dice_detection.init_nn('model_weights_color.pth')

    while True:
        out = DiceItRirect(my_ev3)

        sleep(2)
        toss = GetImage(cam, reader, x, y, w, h, res_w, res_h, model)

        if toss['toss'] == '-':
            print("Could not detect dice.")
        elif toss['toss'] == 'X':
            print("CNN uncertain (probability under 80%).")
            print(toss['toss'], 'time:', toss['time'], 'p =', toss['probability'])
        else:
            print(toss['toss'], 'time:', toss['time'], 'p =', toss['probability'])
        # save img
        save_image(dataset_output_path, toss)

        # db insert
        DB_test.insert_toss(cnx, toss)

        # sleep(0.2)

        # end with whatever key pressed
        if cv2.waitKey(1) != -1:
            break

        pass

        # release all OpenCV thingies
    cam.stop()
    cv2.destroyAllWindows()
    DB_test.close_connection(cnx)

if __name__ == '__main__':
    main()
