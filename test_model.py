import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="fine_tuned_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)

    print(f"Probabilities: {output_data}")
    print(f"Predicted label index: {predicted_index}")
    return predicted_index

# Test with a sample image
predict_image('/Users/mac172/projects/python/dataset/bags/bag-3.jpg')
