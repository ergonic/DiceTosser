#
# Prereqs:
#
# pip install opencv-python
# pip install easyocr
#

import copy
import cv2
import numpy as np
import easyocr
from collections import Counter

# to get reader
def initialize_ocr_reader():
    reader = easyocr.Reader(['en'])

    return reader

def detect_circle_coordinates(image, res_w, res_h):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    r_min = int((120/1920) * res_w)
    r_max = int((220/1920) * res_w)

    # Use the Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=r_min, maxRadius=r_max
    )

    (imgx, imgy, _) = image.shape


    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            if r > r_min and r < r_max:
                try:
                    ret = x - r, y - r, 2 * r, 2 * r
                    # Stop when the detected blob is outside of the image
                    if (x-r > imgx) or (y-r > imgy) or (x-r < 0) or (y-r < 0):
                        return -1, -1, -1, -1

                    return ret  # Return x, y, width, and height of the bounding box
                except:
                    return -1, -1, -1, -1

    # If no circle is found, return None
    return -1,-1,-1,-1

def test_out_rotations(image, reader):
    original = copy.copy(image)

    detections = []
    for rotation in range(0,360,15):
        # grab the dimensions of the image and calculate the center of the rotation
        if rotation == 0:
            rotated = original
        else:
            (h, w) = original.shape[:2]
            (cX, cY) = (w // 2, h // 2)

            M = cv2.getRotationMatrix2D((cX, cY), rotation, 1.0)
            rotated = cv2.warpAffine(original, M, (w, h))

        # try to detect dice face
        result = reader.readtext(rotated)

        # Print results
        for detection in result:
            filtered_text = ''.join(filter(lambda x: 'A' <= x <= 'F', detection[1].upper()))
            if filtered_text != '' and len(filtered_text) == 1:
                #return filtered_text
                detections.append(filtered_text)

    faces_count = Counter(detections).most_common(1)
    if len(faces_count) == 1:
        return faces_count[0][0]

    return None

def get_frame_center(frame, x, y, w, h):

    frame_center = frame[y:y + h, x:x + w]

    return frame_center
