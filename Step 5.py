import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Paths for local machine
base_dir = r'D:\AER850\AER850-Project-2\Project 2 Data\Data'
test_dir = os.path.join(base_dir, 'test')

# Load the trained model
model = load_model(r'D:\AER850\AER850-Project-2\Project_2_Final_Model.h5')

# Test images
test_images = [
    os.path.join(test_dir, 'crack/test_crack.jpg'),
    os.path.join(test_dir, 'missing-head/test_missinghead.jpg'),
    os.path.join(test_dir, 'paint-off/test_paintoff.jpg')
]

# Class labels
class_names = ['crack', 'missing-head', 'paint-off']

# Visualize predictions with probabilities
for img_path in test_images:
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(500, 500))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict probabilities and determine the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Extract class label and probabilities
    predicted_label = class_names[predicted_class]
    probabilities = prediction[0]

    # Create a visualization
    plt.figure()
    plt.imshow(image.load_img(img_path))
    plt.title(f'Predicted: {predicted_label}\nProbabilities: {dict(zip(class_names, probabilities.round(2)))}')
    plt.axis('off')
    plt.show()
