import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
# Define image dimensions and input shape
img_width, img_height = 256, 256
input_shape = (img_width, img_height, 3)
train_data_dir = r"D:\\Btech sem\\project\\Image Morphing\\ela_images"  # Update this path to your actual directory

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    class_mode='binary',
    subset='validation'
)

print(f"Number of training samples: {train_generator.samples}")
print(f"Number of validation samples: {validation_generator.samples}")

if train_generator.samples == 0 or validation_generator.samples == 0:
    raise ValueError("No training or validation samples found. Please check the dataset directory structure.")

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 2
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Ensure the 'models' directory exists
model_save_path = r"C:\\Users\Admin\\OneDrive\Desktop\\Image morphing\\models\\ela_classifier.keras"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
       
# Save the model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")