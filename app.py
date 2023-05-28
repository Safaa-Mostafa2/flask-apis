from __future__ import division, print_function
from flask import Flask,request,jsonify,send_from_directory
from flask_restful import Api,Resource
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
api = Api(app)
model3=pickle.load(open('model.pkl','rb'))

model2 = pickle.load(open('model2.pkl','rb'))
dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)
def convertTuple(tup):
    str = ''
    for item in tup:
        str = str + item
    return str

@app.route('/predictHeart',methods=['post'])
def post():
        posted_data = request.get_json()
        age = posted_data['age']
        sex = posted_data['sex']
        cp = posted_data['cp']
        trestbps = posted_data['trestbps']
        chol = posted_data['chol']
        fbs = posted_data['fbs']
        restecg = posted_data['restecg']
        thalach = posted_data['thalach']
        exang = posted_data['exang']
        oldpeak = posted_data['oldpeak']
        slope = posted_data['slope']
        ca = posted_data['ca']
        thal = posted_data['thal']
        prediction = model3.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        if prediction == 1 :
         prediction_class_en = "You have heart disease, please consult a Doctor."
         prediction_class_ar = "لديك مرض القلب ، يرجى استشارة الطبيب"
        elif prediction == 0 :
           prediction_class_en = "You don't have heart disease.",
           prediction_class_ar = "ليس لديك مرض القلب.",
        else :
           prediction_class_en ="not found" 
           prediction_class_ar ="لا يوجد" 
        if request.headers.get('lan') == 'en':
          return jsonify({
         
           "apiStatus":"true",
           "predicted": convertTuple(prediction_class_en),
          })
        elif request.headers.get('lan') == 'ar':
            return jsonify({
           "apiStatus":"true",
           "predicted": convertTuple(prediction_class_ar),
        })   
@app.route('/predictDiabetes',methods=['post'])
def yy():  
        posted_data = request.get_json()
        Glucose = posted_data['Glucose']
        Insulin = posted_data['Insulin']
        BMI = posted_data['BMI']
        Age = posted_data['Age']
        float_features = [float(Glucose),float(BMI),float(Age),float(Insulin)]
        final_features = [np.array(float_features)]
        prediction = model2.predict( sc.transform(final_features) )
        if prediction == 1 :
           prediction_class_en = "You have Diabetes, please consult a Doctor."
           prediction_class_ar = "لديك مرض السكري ، يرجى استشارة الطبيب"
        elif prediction == 0 :
           prediction_class_en = "You don't have Diabetes.",
           prediction_class_ar = "ليس لديك مرض السكري.",
        else :
           prediction_class_en ="not found" 
           prediction_class_ar ="لا يوجد" 
        if request.headers.get('lan') == 'en':
          return jsonify({
         
           "apiStatus":"true",
           "predicted": convertTuple(prediction_class_en),
          })
        elif request.headers.get('lan') == 'ar':
            return jsonify({
           "apiStatus":"true",
           "predicted": convertTuple(prediction_class_ar),
        })   

if __name__== '__main__':
 app.run(debug=True)
