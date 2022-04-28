from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def load_page():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    a = float(request.form["MinTemp"])
    b = float(request.form["MaxTemp"])
    c = float(request.form["Rainfall"])
    d = float(request.form["Evaporation"])
    e = float(request.form["Sunshine"])
    f = float(request.form["WindGustDir"])
    g = float(request.form["WindGustSpeed"])
    h = float(request.form["WindDir9am"])
    i = float(request.form["WindDir3pm"])
    j = float(request.form["WindSpeed9am"])
    k = float(request.form["WindSpeed3pm"])
    l = float(request.form["Humidity9am"])
    m = float(request.form["Humidity3pm"])
    n = float(request.form["Pressure9am"])
    o = float(request.form["Pressure3pm"])
    p = float(request.form["Cloud9am"])
    q = float(request.form["Cloud3pm"])
    r = float(request.form["Temp9am"])
    s = float(request.form["Temp3pm"])
    t = float(request.form["RainToday"])
    u = float(request.form["month"])
    v = float(request.form["day"])

    x=[[a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v]]

    def Ensemble_Model(x, model1, model2, model3):
        pred = []
        x = np.array(x)
        l1 = model1.predict(x)
        l2 = model2.predict(x)
        l3 = model3.predict(x)
        for i in range(len(l1)):
            pred.append(max([l1[i], l2[i], l3[i]], key=[l1[i], l2[i], l3[i]].count))
        pred = np.array(pred)
        return pred

    if Ensemble_Model(x,model,model1,model2) == 0:
        return render_template('index.html', z='NOT RAIN')
    else:
        return render_template('index.html', z='RAIN')


if __name__ == "__main__":
    app.run(debug=True)
