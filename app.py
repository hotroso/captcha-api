import os
import uuid

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request
from tensorflow import keras
from werkzeug.utils import secure_filename

app = Flask(__name__)
print("Tensorflow version: ", tf.__version__)

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 2 to enable
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# final.captcha.20220831.6000
characters = ['2', '3', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q',
              'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
char_to_labels = {'2': 0, '3': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'A': 7, 'B': 8, 'C': 9, 'D': 10, 'E': 11,
                  'F': 12, 'G': 13, 'H': 14, 'J': 15, 'K': 16, 'M': 17, 'N': 18, 'P': 19, 'Q': 20, 'R': 21, 'S': 22,
                  'T': 23, 'V': 24, 'W': 25, 'X': 26, 'Y': 27, 'Z': 28}
labels_to_char = {0: '2', 1: '3', 2: '5', 3: '6', 4: '7', 5: '8', 6: '9', 7: 'A', 8: 'B', 9: 'C', 10: 'D', 11: 'E',
                  12: 'F', 13: 'G', 14: 'H', 15: 'J', 16: 'K', 17: 'M', 18: 'N', 19: 'P', 20: 'Q', 21: 'R', 22: 'S',
                  23: 'T', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'}

model = keras.models.load_model('./final.captcha.20220831.6000.h5', compile=False)


def xpredict(img_path):
    total_img = 1
    img_height = 40
    img_width = 200

    batch_images = np.ones((total_img, img_width, img_height, 1), dtype=np.float32)

    images = np.zeros((total_img, img_height, img_width), dtype=np.float32)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (img_width, img_height))
    img = (img / 255.).astype(np.float32)

    images[0, :, :] = img

    # 1. Get the image and transpose it
    i_img = img.T
    # 2. Add extra dimenison
    i_img = np.expand_dims(i_img, axis=-1)

    batch_images[0] = i_img
    prediction = model.predict(batch_images)
    pred = prediction[:, :-2]
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred,
                                       input_length=input_len,
                                       greedy=True)[0][0]
    # Iterate over the results and get back the text
    outstr = ''
    for res in results.numpy():
        outstr = ''
        for c in res:
            if len(characters) > c >= 0:
                outstr += labels_to_char[c]
        print('found a text [', outstr, '] in image ', img_path)
    return outstr


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/v1/detect_by_file', methods=['POST'])
def detect_by_file():
    if 'file' not in request.files:
        return {
            "type": "ERR",
            "result": "No file part"
        }
    file = request.files['file']
    if file.filename == '':
        return {
            "type": "ERR",
            "result": "No selected file"
        }
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print('received file ', filename)
        dest_file = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + filename)
        file.save(dest_file)
        str_predicted = xpredict(dest_file)
        os.remove(dest_file)
        return {
            "type": "OK",
            "result": str_predicted
        }
    return {
        "type": "ERR",
        "result": "Unknow error"
    }


@app.route("/")
def hello_world():
    return {
        "type": "OK",
        "result": "you are using my API"
    }


if __name__ == '__main__':
    app.run()
