import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load your saved model
model = tf.keras.models.load_model('digit_recognition.h5')

# Preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (28, 28))
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to make it 4-dimensional
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# Predict the number
def predict_number(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess_image(image)
    prediction = model.predict(image)[0]
    predicted_number = np.argmax(prediction)
    return f"Your written number is: {predicted_number}"

# Create a Gradio interface with a canvas for drawing
iface = gr.Interface(
    fn=predict_number,
    inputs=gr.inputs.Image(shape=(28, 28), image_mode='L', source='canvas'),
    outputs='text',
    title='Handwritten Digit Recognition',
    description='Draw a digit (0-9) on the canvas and see the predicted number.'
)

# Launch the interface
iface.launch()
