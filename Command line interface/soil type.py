import numpy as np
from tensorflow import keras
from keras.preprocessing.image import img_to_array, load_img

# Load the model
model = keras.models.load_model('my_model.h5')

# Specify the path to your input image (photo.jpg)
image_path = 'photo.jpeg'

# Load and preprocess the input image
target_size = (220, 220)
img = load_img(image_path, target_size=target_size)

# Convert the image to a NumPy array and normalize the pixel values
preprocessed_input = img_to_array(img) / 255.0  # Assuming the image is in the range [0, 255]

# Make predictions
predictions = model.predict(np.expand_dims(preprocessed_input, axis=0))

# Find the predicted class (index)
predicted_class = np.argmax(predictions)

# Map the predicted class to the class label
class_labels = ['2', '5', '3', '4', '1']
#yello-sandstone-1
#black-silty-2
#laterite-shale-3
#peat-collivium-4
#cinder-allivium-5
predicted_label = class_labels[predicted_class]

# Print the predicted class label
print(f"Predicted Soil Type: {predicted_label}")

with open("soil_type.txt", "w") as soil_type_file:
    soil_type_file.write(predicted_label)