import tensorflow as tf
print(tf.__version__)

from flask import Flask, request
from create_model import create_model
from PIL import Image
import numpy as np

app = Flask(__name__)

model = create_model()
model.load_weights("./checkpoints/my_checkpoint")

@app.route("/", methods=['POST'])
def isFire():
    file = request.files['file']
    print("Received file")
    img = Image.open(file)
    imResize = img.resize((28, 28), Image.ANTIALIAS).convert('RGB')
    imResize = np.array(imResize)
    imResize = imResize[None, :]
    print(imResize.shape)
    prediction = model.predict(imResize)
    return {'response': int(np.argmax(prediction[0]))}


if __name__ == "__main__":
    print("App running")
    app.run(debug=True)
