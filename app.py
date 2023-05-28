from __future__ import division, print_function
from flask import Flask,request,jsonify,send_from_directory
from flask_restful import Api,Resource
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf
# Flask utils
from werkzeug.utils import secure_filename
from heatmap import save_and_display_gradcam,make_gradcam_heatmap
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
 
 
 
# skin cancer start  





app.config['HEATMAP_FOLDER'] = 'heatmap'
app.config['UPLOAD_FOLDER'] = 'uploads'
# Model saved with Keras model.save()
MODEL_PATH = './model_v1.h5'


#Load your trained model
model = load_model(MODEL_PATH)



class_dict = {0:"Basal_Cell_Carcinoma (Cancer)",
             1:"Melanoma (Cancer)",
             2:"Nevus (Non-Cancerous)"}

@app.route('/uploads/<filename>')
def upload_img(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        


def model_predict(img_path, model):
    img = Image.open(img_path).resize((224,224)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
    preds = model.predict(img)[0]
    prediction = sorted(
      [(class_dict[i], round(j*100, 2)) for i, j in enumerate(preds)],
      reverse=True,
      key=lambda x: x[1]
  )
    return prediction,img


@app.route('/predict', methods=[ 'POST'])
def predict(): 

        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        file_name=os.path.basename(file_path)
        # Make prediction
        pred,img = model_predict(file_path, model)
        last_conv_layer_name = "block_16_depthwise"
        heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)
        fname=save_and_display_gradcam(file_path, heatmap)

        classA = pred[0][0]
        classB = pred[1][0]
        classC = pred[2][0]
        return jsonify({
           "apiStatus":"true",
           classA:pred[0][1],
           classB:pred[1][1],
           classC:pred[2][1]
         # preds:pred
          })

 
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