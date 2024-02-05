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
import torch
from nn import SimpleCNN
import torchvision.transforms as transforms

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


def init_nn(model_path):

    # Create the model instance
    model = SimpleCNN(num_classes=6)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the saved state dict
    model.load_state_dict(torch.load(model_path, map_location=device))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    return model


def get_nn_label(model, dice_image):

    # Define a transform to convert
    # the image to torch tensor
    # Convert to PyTorch tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4798, 0.4511, 0.4503], [0.1495, 0.1556, 0.1532])
    ])
    input_tensor = transform(dice_image)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    # Make a prediction
    with torch.no_grad():
        prediction = model(input_tensor)

    # Process the prediction as needed (e.g., apply softmax if required)
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(prediction)
    print("Prediction:", prediction)
    print("Probs:", probabilities)
    predicted_class = torch.argmax(probabilities, dim=1)

    # Output the predicted class
    print("Predicted class:", predicted_class.item())



