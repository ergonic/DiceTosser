import os
import cv2
from torchvision.datasets import ImageFolder


def convert_to_grayscale_and_save_opencv(root_dir, target_dir):
    """
    Convert all images in the directory structure of root_dir to grayscale using OpenCV and save them in a
    new directory structure under target_dir, preserving the original directory structure.

    Parameters:
    - root_dir (str): Root directory of the original images.
    - target_dir (str): Root directory where grayscale images will be stored.
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Load dataset with ImageFolder to leverage the directory structure
    dataset = ImageFolder(root=root_dir)

    for idx, _ in enumerate(dataset):
        # Get the path of the current image
        image_path = dataset.samples[idx][0]
        # Create a relative path for the image to preserve the directory structure
        relative_path = os.path.relpath(image_path, root_dir)
        # Define the target path for the grayscale image
        target_path = os.path.join(target_dir, relative_path)
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Read the image using OpenCV
        image = cv2.imread(image_path)
        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_image = cv2.equalizeHist(grayscale_image)

        # Save the grayscale image
        cv2.imwrite(target_path, grayscale_image)

    print(f"Conversion complete. Grayscale images are stored in '{target_dir}'.")


# Example usage
root_dir = os.path.join('D:','dicetoss_clean_test')  # Change this to your dataset's root directory
target_dir = os.path.join('D:','dicetoss_grayscale')  # Change this to where you want to store grayscale images
convert_to_grayscale_and_save_opencv(root_dir, target_dir)
