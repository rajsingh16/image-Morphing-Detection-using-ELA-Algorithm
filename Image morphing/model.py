from glob import glob
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image, ImageChops, ImageEnhance
from matplotlib import pyplot as plt
from tensorflow import keras

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

def ela_image(input_image_path, output_image_path, quality=95):
    original = cv2.imread(input_image_path)
    if original is None:
        print(f"Failed to load image {input_image_path}")
        return

    temp_filename = "temp_ela_image.jpg"
    cv2.imwrite(temp_filename, original, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    compressed = cv2.imread(temp_filename)
    if compressed is None:
        print(f"Failed to load image {temp_filename}")
        return

    difference = cv2.absdiff(original, compressed)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    mean_diff = np.mean(gray_difference)
    scale = 100.0 / mean_diff if mean_diff != 0 else 1
    enhanced_difference = cv2.convertScaleAbs(gray_difference, alpha=scale, beta=0)
    cv2.imwrite(output_image_path, enhanced_difference)
    os.remove(temp_filename)


def generate_ela_images(input_dir, output_dir, quality=95):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in ['Au', 'Tp']:
        input_category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        
        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)
        
        for img_name in os.listdir(input_category_dir):
            input_img_path = os.path.join(input_category_dir, img_name)
            output_img_path = os.path.join(output_category_dir, img_name)
            ela_image(input_img_path, output_img_path, quality)

input_dir = r"D:\\Btech sem\\project\\Image Morphing\\CASIA2"
output_dir = "D:\\Btech sem\\project\\Image Morphing\\ela_images"
generate_ela_images(input_dir, output_dir)
