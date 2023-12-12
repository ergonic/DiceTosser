import cv2
import ev3_dc as ev3
from time import sleep
import dice_detection


# my_ev3 = ev3.EV3(
#     protocol=ev3.USB,
#     host="00:16:53:47:3A:00",
#     verbosity=0
# )
# my_ev3.sync_mode = ev3.SYNC

def DiceIt():
    ops = b''.join((
        ev3.opFile,
        ev3.LOAD_IMAGE,
        ev3.LCX(1),  # SLOT
        ev3.LCS('/home/root/lms2012/prjs/Dicing30/Dicing30.rbf'),  # NAME
        ev3.LVX(0),  # SIZE
        ev3.LVX(4),  # IP*
        ev3.opProgram_Start,
        ev3.LCX(1),  # SLOT
        ev3.LVX(0),  # SIZE
        ev3.LVX(4),  # IP*
        ev3.LCX(0)  # DEBUG
    ))
    my_ev3.send_direct_cmd(ops, local_mem=8)
    pass


def GetImage():
    ret, frame = cam.read()
    center = dice_detection.get_frame_center(frame, x, y, w, h)
    dx, dy, dw, dh = dice_detection.detect_circle_coordinates(center)
    if dx == -1:
        return None
    dice = center[dy:dy + dh, dx:dx + dw]
    cv2.rectangle(frame, (dx + x, dy + y), (dx + x + dw, dy + y + dh), (255, 0, 0), 2)
    cv2.putText(frame, 'DICE', (dx + x, dy + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    detected_letter = dice_detection.test_out_rotations(dice, reader)
    cv2.putText(frame, detected_letter, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Dice detection", frame)
    return detected_letter


# count = 0
# while(count < 1):
#
#     DiceIt()
#     sleep(1)
#     ret, frame = cam.read()
#
#     center = dice_detection.get_frame_center(frame, x, y, w, h)
#     dx,dy,dw,dh = dice_detection.detect_circle_coordinates(center)
#     if dx == -1:
#         continue
#     else:
#         dice = center[dy:dy + dh, dx:dx + dw]
#         cv2.rectangle(frame, (dx + x, dy + y), (dx + x + dw, dy + y + dh), (255, 0, 0), 2)
#         cv2.putText(frame, 'DICE', (dx + x, dy + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#         detected_letter = dice_detection.test_out_rotations(dice, reader)
#         cv2.putText(frame, detected_letter, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#
#     cv2.imshow("Dice detection", frame)
#     cv2.waitKey()
#
#     break
#
#     count += 1


cam = None
reader = None
x, y, w, h = [None, None, None, None]

if __name__ == '__main__':
    # connector to EV3 brick
    my_ev3 = ev3.EV3(protocol=ev3.USB)

    # set variables for openCV and OCR
    cam = cv2.VideoCapture(0)
    reader = dice_detection.initialize_ocr_reader()
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    x = 800
    y = 400
    w = 450
    h = 450

    while True:
        sleep(1)
        DiceIt()
        letter = GetImage()
        print(letter)

        # end with whatever key pressed
        if cv2.waitKey(1) != -1:
            break

        pass

    # release all OpenCV thingies
    cam.release()
    cv2.destroyAllWindows()
