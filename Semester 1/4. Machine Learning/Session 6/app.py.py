# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:08:44 2021
@author: Senthil

Updated on Tue Jan  21 13:17:44 2021
@author: kaustuv Kunal

"""



import flask
import pickle
import pandas as pd

my_model=pickle.load(open('full_pipeline','rb'))

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        Married = flask.request.form['Married']
        Education = flask.request.form['Education']
        ApplicantIncome = flask.request.form['ApplicantIncome']
        LoanAmount = flask.request.form['LoanAmount']
        Credit_History = flask.request.form['Credit_History']

        # Make DataFrame for model
        test = pd.DataFrame([[Married,Education,ApplicantIncome,LoanAmount,Credit_History]],
                                       columns=['Married','Education','ApplicantIncome','LoanAmount','Credit_History'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction

        prediction=my_model.predict(test)[0]
        
        
        #prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Married':Married,
                                                     'Education':Education,
                                                     'ApplicantIncome':ApplicantIncome,
                                                     'LoanAmount':LoanAmount,
                                                     'Credit_History':Credit_History},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()
