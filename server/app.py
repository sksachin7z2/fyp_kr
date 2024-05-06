
import json
from flask import Flask, url_for, request, render_template,redirect,jsonify,make_response,send_from_directory
import numpy as np
from labels import labels
from tensorflow.keras.models import load_model
import numpy as np
import sys
from PIL import Image
from flask_cors import CORS

sys.modules['Image'] = Image 

# Declare a flask app
app = Flask(__name__)
CORS(app)



# Model saved with Keras model.save()

SPEAKER_CLASSIFIER_MODEL_PATH = 'model/modelimage.h5'

model=load_model(SPEAKER_CLASSIFIER_MODEL_PATH)
# model1.summary()
print('Models loaded')


ALLOWED_EXTENSIONS = {'webp', 'jpg','jpeg','png'}  # Add more file extensions as needed

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API endpoint for handling audio data


def preprocess_image(image_path):
    # Load the image and resize it to (224,224)
    img = Image.open(image_path)
    img = img.resize((224, 224))
    # Convert image to RGB if it's not
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Convert PIL image to numpy array
    img_array = np.array(img)
    # Normalize the image to be in the range [0, 1]
    # img_array = img_array / 255.0
    # Expand the dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    # Make prediction
    # print(img_array,img_array.shape)
    preds = model.predict(img_array)
    # preds = softmax(preds, axis=-1)
    ind=np.argmax(preds)
    # print(y_test[ind],preds)
    # print(ind,preds)
    top_indices = np.argsort(preds[0])[::-1][:5]

    # Initialize an empty dictionary to store the results
    top_predictions_dict = {}

    # Loop through the top indices and populate the dictionary
    for i, index in enumerate(top_indices):
        class_label = labels[index]
        probability = preds[0][index]

        # Add the class and probability to the dictionary
        top_predictions_dict[class_label] =np.float32(probability)
    serializable_dict = {key: float(value) for key, value in top_predictions_dict.items()}

    # Serialize the dictionary to JSON
    json_data = json.dumps(serializable_dict)
    print(json_data)
    # Print the top predictions and their probabilities
    # print(f"Top {i+1}: Class {class_label} - Probability: {probability}")
    return {'top':labels[ind],'pred':json_data}
    
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Securely save the file in the UPLOAD_FOLDER directory
        # filepath = os.path.join(app.config['UPLOAD_FOLDER'], "audio.wav")
        image_path=f"image.{file.filename.rsplit('.', 1)[1].lower()}"
        file.save(image_path)
        pred=predict_image(image_path)

        # print(image.shape)

        return jsonify({'message': 'File uploaded successfully','pred':pred['top'],'prob':pred['pred']})

    return jsonify({'error': 'Invalid file extension'})

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0')