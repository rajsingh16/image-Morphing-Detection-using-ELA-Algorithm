import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageChops, ImageEnhance

# Define the function to create ELA image
def ela_image(image_path, output_path, quality=90):
    original = Image.open(image_path)
    original.save(output_path, 'JPEG', quality=quality)
    temporary = Image.open(output_path)
    
    diff = ImageChops.difference(original, temporary)
    
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    
    diff.save(output_path)

try:
    model = load_model(r'C:\\Users\\Admin\\OneDrive\\Desktop\\Image morphing\\models\\ela_classifier.keras')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)


# Print the model summary to inspect the input shape
model.summary()

# Get the input shape from the model
input_shape = model.input_shape[1:]
print(f"Model expects input shape: {input_shape}")

# Extract the width and height from the input shape
img_width, img_height = input_shape[0], input_shape[1]


# Load the image and resize it to the expected size
img = image.load_img(r"D:\\Btech sem\\project\\Image Morphing\\Original Images\\Image_15.jpg", target_size=(256, 256))
img_array = image.img_to_array(img) / 255.0  # Normalize the image

# Ensure the array has the right shape (1, 256, 256, 3)
img_array = np.expand_dims(img_array, axis=0)


def resize_image(image_path, target_size=(256, 256)):
    """Resizes the image to the target size while maintaining three color channels."""
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure the image has three color channels (RGB)
    img = img.resize(target_size)
    return img

def predict_edited(image_path):
    ela_image_path = 'temp_ela.jpg'
    ela_image(image_path, ela_image_path)

    img = resize_image(ela_image_path, target_size=(img_width, img_height))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    print(f"Image array shape: {img_array.shape}")

    prediction = model.predict(img_array)
    os.remove(ela_image_path)
    
    print(prediction)
    
    if prediction < 0.5:
        print("The image is not edited.")
    else:
        print("The image is edited.")

# Example prediction
predict_edited(r"D:\\Btech sem\\project\\Image Morphing\\Original Images\\Image_15.jpg")
