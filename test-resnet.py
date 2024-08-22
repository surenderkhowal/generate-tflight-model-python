import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('resnet50_trained_model.keras')

# Define parameters
img_height, img_width = 224, 224  # Same as during training

# Load and preprocess a single image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    return img_array

# Define the path to the single image
test_image_path = '/Users/mac172/projects/python/dataset/test/bags/bag-7.jpeg'  # Update this to your image path
test_image = load_and_preprocess_image(test_image_path)

# Make a prediction
predictions = model.predict(test_image)

# Convert predictions to class labels
predicted_class = np.argmax(predictions, axis=1)

# Get class labels from the model's class indices
# Use the original class indices from training
class_labels = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6']  # Update this based on your training

# Get the predicted label
predicted_label = class_labels[predicted_class[0]]  # Get the predicted label

print(f'Predicted class: {predicted_label}')
