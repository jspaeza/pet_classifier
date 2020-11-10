

from flask import Flask, render_template, request
from fastai.vision.all import *
import pathlib


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

path = Path()
learn = load_learner(path/'export.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            f.save(os.path.join('images/', f.filename))
            img = 'images/' + f.filename

        pred, pred_idx, probs = learn.predict(img)

    return render_template('index.html', prediction_text='Your Prediction :  {} '.format(pred), probs=probs)