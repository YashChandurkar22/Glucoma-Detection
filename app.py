from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')


# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.
    return img_array


class_names = ["no", "yes"]


# Define a function to make a prediction
def predict(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    # return str(prediction)
    return str(class_names[np.argmax(prediction)])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        image_file = request.files['image']
        image_path = 'static/images/' + image_file.filename
        image_file.save(image_path)
        prediction = predict(image_path)
        return render_template('index.html', prediction=prediction, image_path=image_path)


if __name__ == '__main__':
    app.run(debug=True)
