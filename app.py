from unicodedata import name
from flask import Flask, jsonify, render_template, request, url_for, redirect
import numpy as np
import sklearn.externals
import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
#from sklearn.externals import joblib
from bs4 import BeautifulSoup


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/success/<name>')
def result(name):
    return 'Result : %s' % name


@app.route('/predict', methods=['POST', 'GET'] )
def predict():
    clf = joblib.load('model-2.pkl')
    std_scalar_X = joblib.load('std_scaler_X.pkl')
    std_scalar_y = joblib.load('std_scaler_y.pkl')
    if request.method == 'POST':
        value_1=request.form['value_1']
        value_2=request.form['value_2']
        value_3=request.form['value_3']
        value_4=request.form['value_4']
        value_5=request.form['value_5']
        value_6=request.form['value_6']
        value_7=request.form['value_7']
        value_8=request.form['value_8']
        value_9=request.form['value_9']
        value_10=request.form['value_10']
        value_11=request.form['value_11']
        value_12=request.form['value_12']
        import pandas as pd
 
        data = [{"value_10":value_10, "value_9":value_9, "value_8":value_8, "value_7":value_7, "value_6":value_6, "value_5":value_5, "value_4":value_4, "value_3":value_3, "value_2":value_2, "value_1":value_1, "value_12":value_12, "value_11":value_11}]
        
        df = pd.DataFrame(data)
        
        
        y_pred = clf.predict(std_scalar_X.transform(df))
        print("SHAPE--------------------------->")
        print(y_pred.shape)
        #y_pred = std_scalar_y.inverse_transform(y_pred)
        return  redirect(url_for('result', name= y_pred[0][0]))
    else:
        user=request.args.get('nm')
        return redirect(url_for('result', name= y_pred[0][0]))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)