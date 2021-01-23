from flask import Flask, render_template,request, jsonify

import pandas as pd
import pickle 
import numpy as np

app = Flask(__name__,template_folder='templates')
smote = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pred',methods=['POST','GET'])
def pred():
    return render_template('predict.html')

@app.route('/predict',methods=['POST','GET'])
def prediction():
	#a=request.form.values()
	int_features = [int(a) for a in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = smote.predict(final_features)
	if prediction==0:
		op='not Churned'
	else:
		op='Churned'

    #output = round(prediction[0])
	return render_template('predict.html', prediction_text='User has  {}'.format(op))

if __name__ == '__main__':
    app.run(debug=True)