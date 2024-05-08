
from flask import Flask, render_template, request, redirect, url_for, session,jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import tensorflow as tf

import cv2
import numpy as np
import os
import io
import base64

app = Flask(__name__)




# Function to calculate prediction accuracy
def calculate_accuracy(prediction, true_label_index):
    predicted_class_index = prediction.argmax(axis=-1)
    return predicted_class_index[0] == true_label_index

@app.route('/home', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        return render_template('index.html')
    return render_template('index.html')



app.secret_key = 'xyzsdfg'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'login_system'
  
mysql = MySQL(app)
  
@app.route('/')
@app.route('/login', methods =['GET', 'POST'])
def login():
    mesage = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s AND password = % s', (email, password, ))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['name']
            session['email'] = user['email']
            mesage = 'Logged in successfully !'
            return render_template('index.html', mesage = mesage)
        else:
            mesage = 'Please enter correct email / password !'
    return render_template('login.html', mesage = mesage)
  


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/register', methods =['GET', 'POST'])
def register():
    mesage = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form :
        userName = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            mesage = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address !'
        elif not userName or not password or not email:
            mesage = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO user VALUES (NULL, % s, % s, % s)', (userName, email, password, ))
            mysql.connection.commit()
            mesage = 'You have successfully registered !'
    elif request.method == 'POST':
        mesage = 'Please fill out the form !'
    return render_template('register.html', mesage = mesage)






model = tf.keras.models.load_model('model2.h5')

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
        print("Predicted Class Index:", predicted_class_index)
        print("Predicted Probability:", prediction[0][predicted_class_index[0]])
        
        # Calculate prediction accuracy
          # Calculate prediction accuracy percentage
        accuracy_percentage = prediction[0][predicted_class_index[0]] 
        

 
        # Pass the image data and prediction to the HTML template
        return render_template('covid.html', image_data=base64.b64encode(image_data).decode('utf-8'), prediction=classification,accuracy=accuracy_percentage)

    # If it's a GET request, just render the HTML template without passing any image data or prediction
    return render_template('covid.html', image_data=None, prediction=None,accuracy=None)

# Define function to load the tumor detection model
# def load_tumor_model():
#     return load_model('model2.h5')


from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import io
import base64

# Define your class_names list here
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load your trained model
tumor_model = load_model('brain_tumor_model.h5')

# Function to preprocess and predict the image
def load_and_pred_image(filename, img_shape=200):
    # Read in the image
    img = tf.io.read_file(filename)
    
    # Decode the read file into tensor
    img = tf.image.decode_image(img)
    
    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    
    # Rescale the image
    img = img / 255.0
    
    return img

def pred_and_plot(model, filename, class_names=class_names):
    # Import the target image
    img = load_and_pred_image(filename)
    
    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    # Convert the prediction probabilities to class labels
    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]
    
    return pred_class

@app.route('/tumor', methods=['GET', 'POST'])
def tumor():
    if request.method == 'POST':
        # Get the image file from the POST request
        imagefile = request.files['imagefile']

        # Read the image file
        filename = imagefile.filename

        # Get the prediction
        prediction = pred_and_plot(tumor_model, filename)

        # Pass the image data and prediction to the HTML template
        return render_template('brain_tumor.html', image_data=base64.b64encode(imagefile.read()).decode('utf-8'), prediction=prediction)

    # If it's a GET request, just render the HTML template without passing any image data or prediction
    return render_template('brain_tumor.html', image_data=None, prediction=None)

























# tumor_model=load_model('brain_tumor_model.h5')
# #brain tumor detection



# @app.route('/tumor', methods=['GET', 'POST'])
# def tumor():
#     # tumor_model=load_tumor_model()
#     if request.method == 'POST':
#         # Get the image file from the POST request
#         imagefile = request.files['imagefile']

#         # Read the image file
#         image_data = imagefile.read()
        
#         # Load the image from memory
#         image = load_img(io.BytesIO(image_data), target_size=(200, 200))
#         image = img_to_array(image)
#         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#         image = image / 255.0
#         image = preprocess_input(image)

#         # Use your custom model for prediction
#         prediction = tumor_model.predict(image)

#         # Convert the prediction probabilities to class labels
#         class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
#         predicted_class_index = prediction.argmax(axis=-1)
#         classification = class_labels[predicted_class_index[0]]
        
#         print("Predicted Class Index:", predicted_class_index)
#         print("Predicted Probability:", prediction[0][predicted_class_index[0]])
        

#         # Pass the image data and prediction to the HTML template
#         return render_template('brain_tumor.html', image_data=base64.b64encode(image_data).decode('utf-8'), prediction=classification)

#     # If it's a GET request, just render the HTML template without passing any image data or prediction
#     return render_template('brain_tumor.html', image_data=None, prediction=None)






# reno_model = load_model("renopathy_model_first.h5")
#eye renopathy
@app.route('/reno', methods=['GET', 'POST'])
def reno():
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
        return render_template('eye_renopathy.html', image_data=base64.b64encode(image_data).decode('utf-8'), prediction=classification)

    # If it's a GET request, just render the HTML template without passing any image data or prediction
    return render_template('eye_renopathy.html', image_data=None, prediction=None)






#Xray Checkup
# Define your class_names list here
class_names_xray = ['COVID19', 'NORMAL' ,'PNEUMONIA', 'TURBERCULOSIS']

# Load your trained model
Xray_model = load_model('tuberclosis.h5')

#create a function to import and  image and resize it to be able to
def load_and_pred_image_xray(filename,img_shape=224):
   # Read in the image
    img = tf.io.read_file(filename)
    # Decode the read file into tensor
    img = tf.image.decode_image(img, channels=3)  # Specify the number of channels (RGB)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # Rescale the image
    img = img / 255.
    return img


def pred_and_plot_xray(model,filename,class_names=class_names_xray):
       # Load and preprocess the image
    img = load_and_pred_image_xray(filename)
    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    # Adding logic for multiclass to make 1 array
    if len(pred[0]) > 1:
        pred_index = tf.argmax(pred[0])
        pred_class = class_names[pred_index]
        pred_confidence = tf.reduce_max(pred[0])
    else:
        pred_class = class_names[int(pred[0][0])]
        pred_confidence = pred[0][0]
    return pred_class, pred_confidence

@app.route('/xray', methods=['GET', 'POST'])
def xray():
    if request.method == 'POST':
        # Get the image file from the POST request
        imagefile = request.files['imagefile']

        # Read the image file
        filename = imagefile.filename

        # Get the prediction
        prediction,pred_confidence = pred_and_plot_xray(Xray_model, filename)

        # Pass the image data and prediction to the HTML template
        return render_template('Xray.html', image_data=base64.b64encode(imagefile.read()).decode('utf-8'), prediction=prediction,accuracy=pred_confidence)

    # If it's a GET request, just render the HTML template without passing any image data or prediction
    return render_template('Xray.html', image_data=None, prediction=None,accuracy=None)




#pneumonia checkup


# Define your class_names list here
class_names_xray_pneumonia = ['COVID19', 'NORMAL' ,'PNEUMONIA-bacterial', 'pneumonia viral']

# Load your trained model
xray_pneumonia = load_model('brain_tumor_model.h5')

#create a function to import and  image and resize it to be able to
def load_and_pred_image_pnemo(filename,img_shape=200):
   # Read in the image
    img = tf.io.read_file(filename)
    # Decode the read file into tensor
    img = tf.image.decode_image(img, channels=3)  # Specify the number of channels (RGB)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # Rescale the image
    img = img / 255.
    return img


def pred_and_plot_pnemo(model,filename,class_names=class_names_xray_pneumonia):
       # Load and preprocess the image
    img = load_and_pred_image_pnemo(filename)
    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    # Adding logic for multiclass to make 1 array
    if len(pred[0]) > 1:
        pred_index = tf.argmax(pred[0])
        pred_class = class_names[pred_index]
        pred_confidence = tf.reduce_max(pred[0])
    else:
        pred_class = class_names[int(pred[0][0])]
        pred_confidence = pred[0][0]
    return pred_class, pred_confidence


@app.route('/pnemo', methods=['GET', 'POST'])
def pnemo():
    if request.method == 'POST':
        # Get the image file from the POST request
        imagefile = request.files['imagefile']

        # Read the image file
        filename = imagefile.filename

        # Get the prediction
        prediction,pred_confidence = pred_and_plot_pnemo(xray_pneumonia, filename)

        # Pass the image data and prediction to the HTML template
        return render_template('Pneumonia.html', image_data=base64.b64encode(imagefile.read()).decode('utf-8'), prediction=prediction,accuracy=pred_confidence)

    # If it's a GET request, just render the HTML template without passing any image data or prediction
    return render_template('Pneumonia.html', image_data=None, prediction=None,accuracy=None)



#Brain tumor segmentation
# @app.route('/segment', methods=['GET','POST'])
# def segment():
#     if request.method == 'POST':
#         return render_template('segment.html')
#     return render_template('segment.html')

# Load the segmentation model
from unet_model import build_unet
model_2 = build_unet((256,256,1))
model_2.load_weights('new_train_rgb.h5')

import numpy as np
import cv2
from tensorflow.keras.utils import normalize


def preprocess_segmentation_image(image_buffer, target_size=(256, 256)):
   test_img_other = cv2.imread(image_buffer,0)
   test_img_other_resize=cv2.resize(test_img_other, (256, 256))
   test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other_resize),axis=1),2)
   test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
   test_img_other_input = np.expand_dims(test_img_other_norm,0)
   return test_img_other_input



@app.route('/segment', methods=['GET', 'POST'])
def segment():
    if request.method == 'POST':
        imagefile = request.files.get('imagefile')
        if imagefile:
            # Read the image file to buffer
            filename = imagefile.filename
            # filestr = imagefile.read()
            # npimg = np.frombuffer(filestr, np.uint8)
            # Preprocess the image
            prediction_img = preprocess_segmentation_image(filename)
            # Predict using the model
            prediction_other = (model_2.predict(prediction_img)[0,:,:,0] > 0.50).astype(np.uint8)

            # Convert the prediction to an image
            pred_img = (prediction_other * 255).astype(np.uint8)
            _, encoded_img = cv2.imencode('.PNG', pred_img)
            encoded_img = base64.b64encode(encoded_img).decode('utf-8')

            # Render the HTML page with the processed image
            return render_template('segment.html', segmented_image=encoded_img)

    # If it's a GET request or no file was posted, render the page normally
    return render_template('segment.html', segmented_image=None)




# def preprocess_segmentation_image(filename, target_size=(256, 256)):
#     test_img_other = cv2.imread(filename,0)
#     test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other),axis=1),2)
#     test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
#     test_img_other_input = np.expand_dims(test_img_other_norm,0)
#     return test_img_other_input
    


# @app.route('/segment', methods=['GET', 'POST'])
# def segment():
#     if request.method == 'POST':
      

#                 imagefile = request.files['imagefile']
#                 filestr = imagefile.read()
#                 npimg = np.frombuffer(filestr, np.uint8)
#                 prediction_img= preprocess_segmentation_image(npimg)
#                 # Predict using the model
#                 prediction_other = (model.predict(prediction_img)[0,:,:,0]>0.10).astype(np.uint8)

#                 # Convert the prediction to an image
#                 pred_img = (prediction * 255).astype(np.uint8)
#                 _, encoded_img = cv2.imencode('.PNG', pred_img)
#                 encoded_img = base64.b64encode(encoded_img).decode('utf-8')

#                 # Render the HTML page with the processed image
#                 return render_template('segment.html', segmented_image=encoded_img)
#     # If it's a GET request or no file was posted, render the page normally
#     return render_template('segment.html', segmented_image=None)



if __name__ == '__main__':
    app.run(port=3000, debug=True)
