from wsgiref import simple_server

from flask import Flask, request, app
from flask import Response
from flask_cors import CORS,cross_origin
from flask import Flask, render_template, request,jsonify
from Data_prep_and_pred.Data_prep_and_pred import batch_data_pred
from Data_prep_and_pred.Data_prep_and_pred import single_pred
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__) # initializing a flask app
app=application
CORS(app)
app.config['DEBUG'] = True


#class ClientApi:

 #   def __init__(self):
  #      self.batch_data_pred = batch_data_pred()
   #     self.single_pred = single_pred()

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict_batch', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def predict_batch():
    if request.method == 'POST':
        try:
            df = pd.read_csv(request.files["file"])

            if 'Unnamed: 0' in df.columns:
                test_df = df.drop(['Unnamed: 0'], axis=1)
            else:
                test_df = df


            batch_data_prep_object = batch_data_pred(test_df)
            dataframe_to_impute = batch_data_prep_object.nan_feature(test_df)
            imputed_dataframe = batch_data_prep_object.impute_data(test_df, dataframe_to_impute)
            test_df[['Glucose',
                        'BloodPressure',
                        'SkinThickness',
                        'Insulin',
                        'BMI',
                        'DiabetesPedigreeFunction']] = imputed_dataframe

            X = test_df.drop(['Outcome'], axis=1)
            scaler_transformed_dataframe = batch_data_prep_object.scaler_transform(X)
            y_pred = batch_data_prep_object.batch_predict(scaler_transformed_dataframe)
            prediction = pd.Series(y_pred)
            prediction = list(prediction.map({1: 'Diabetic', 0: 'Non - Diabetic'}))
            print(prediction)
            return render_template('results.html', prediction=prediction)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


@app.route('/predict_single',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def predict_single():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Pregnancies=float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])

            single_pred_obj = single_pred()
            data_pred = {'Pregnancies':Pregnancies,'Glucose':Glucose,'BloodPressure':BloodPressure,'SkinThickness':SkinThickness,'Insulin':Insulin,'BMI':BMI,'DiabetesPedigreeFunction':DiabetesPedigreeFunction,'Age':Age}
            prediction = single_pred_obj.single_predict(data_pred)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=prediction)

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


@app.route("/from_postman_single", methods=['POST'])
def from_postman_single():
    try:
        if request.json['data'] is not None:
            data = request.json['data']
            print('data is:     ', data)
            single_pred_obj = single_pred()
            res = single_pred_obj.single_predict(data)

            #result = clntApp.predObj.predict_log(data)
            print('result is        ',res)
            return Response(res)
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ',e)
        return Response(e)


if __name__ == "__main__":
    #clntApp = ClientApi()
    host = '0.0.0.0'
    port = 5000
    app.run(debug=True)
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()