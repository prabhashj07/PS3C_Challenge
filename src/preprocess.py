import cv2 
import numpy as np 

def transform_image(image, padding=50, target_size=(224,224)):
    """
    Adds padding to an image and then downsample it. 

    :param image: Input image as a Numpy array. 
    :param padding: Amount of padding to add.
    :param target_size: Desired size of the output image (width, height).
    :return: Transformed image.
    """
    # Get the image dimensions
    h, w = image.shape[:2]

    # Add padding
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0,0,0))

    # Downscale the image 
    resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_AREA)

    return resized_image