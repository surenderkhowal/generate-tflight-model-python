import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Define parameters
img_height, img_width = 180, 180
batch_size = 32

# Directory where your test data is located
test_data_dir = 'dataset/test'

# Load test dataset
test_dataset = image_dataset_from_directory(
    test_data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# Normalization function
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply normalization
test_dataset = test_dataset.map(preprocess)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="fine_tuned_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize variables to keep track of metrics
correct_predictions = 0
total_predictions = 0

# Iterate over the test dataset
for images, labels in test_dataset:
    # Convert labels to numpy array
    labels = labels.numpy()

    # Loop over each image in the batch
    for i in range(images.shape[0]):
        # Get the image
        image = images[i]

        # Expand dimensions to match the input shape of the model
        input_data = np.expand_dims(image, axis=0)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted class
        predicted_class = np.argmax(output_data, axis=1)

        # Compare with true label
        if predicted_class == labels[i]:
            correct_predictions += 1
        total_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print(f'Test Accuracy: {accuracy * 100:.2f}%')
