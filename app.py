from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import os
import io
import base64

app = Flask(__name__)

# Load your custom model from the .h5 file
model = load_model('model2.h5')
@app.route('/index', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        return render_template('index.html')
    return render_template('index.html')

@app.route('/covid', methods=['GET', 'POST'])
def covid():
    if request.method == 'POST':
        # Get the image file from the POST request
        imagefile = request.files['imagefile']

        # Read the image file
        image_data = imagefile.read()

        # Load the image from memory
        image = load_img(io.BytesIO(image_data), target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Use your custom model for prediction
        prediction = model.predict(image)

        # Convert the prediction probabilities to class labels
        class_labels = ['covid virus', 'normal', 'others_virus']
        predicted_class_index = prediction.argmax(axis=-1)
        classification = class_labels[predicted_class_index[0]]

        # Pass the image data and prediction to the HTML template
        return render_template('covid.html', image_data=base64.b64encode(image_data).decode('utf-8'), prediction=classification)

    # If it's a GET request, just render the HTML template without passing any image data or prediction
    return render_template('covid.html', image_data=None, prediction=None)


#brain tumor detection
@app.route('/tumor', methods=['GET', 'POST'])
def tumor():
    if request.method == 'POST':
        # Get the image file from the POST request
        imagefile = request.files['imagefile']

        # Read the image file
        image_data = imagefile.read()

        # Load the image from memory
        image = load_img(io.BytesIO(image_data), target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Use your custom model for prediction
        prediction = model.predict(image)

        # Convert the prediction probabilities to class labels
        class_labels = ['covid virus', 'normal', 'others_virus']
        predicted_class_index = prediction.argmax(axis=-1)
        classification = class_labels[predicted_class_index[0]]

        # Pass the image data and prediction to the HTML template
        return render_template('brain_tumor.html', image_data=base64.b64encode(image_data).decode('utf-8'), prediction=classification)

    # If it's a GET request, just render the HTML template without passing any image data or prediction
    return render_template('brain_tumor.html', image_data=None, prediction=None)





if __name__ == '__main__':
    app.run(port=3000, debug=True)
