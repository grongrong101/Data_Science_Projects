# minimal example from:
# http://flask.pocoo.org/docs/quickstart/

import pickle
import numpy as np
import flask
from flask import render_template, request, Flask

app = Flask(__name__)  # create instance of Flask class

repeat_doge = 3

with open("lr.pkl", "rb") as f:
    lr_model = pickle.load(f)

@app.route('/')  # the site to route to, index/main in this case
def magical_image() -> str:
    return '<html>' + repeat_doge * '<img src="/static/doge.jpg">' + '</html>'

@app.route('/predict')
def predict():
    return str(lr_model.feature_names)

@app.route('/predict2')
def predict2():
    return render_template('predictor.html',
                           feature_names=lr_model.feature_names)

@app.route('/predict3')
def predict3():
    return render_template('predictor2.html',
                           feature_names=lr_model.feature_names)

@app.route("/predict_final", methods=["POST", "GET"])
def predict_final():

    x_input = []
    for i in range(len(lr_model.feature_names)):
        f_value = float(
            request.args.get(lr_model.feature_names[i], "0")
            )
        x_input.append(f_value)

    pred_probs = lr_model.predict_proba([x_input]).flat

    return flask.render_template('predict_final.html',
    feature_names=lr_model.feature_names,
    x_input=x_input,
    prediction=list(np.argsort(pred_probs)[::-1])
    )

if __name__ == '__main__':
    app.run()
