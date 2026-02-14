#==================
#Import Libraries
#==================

import numpy as np
import pandas as pd
import gradio as gr
import pickle

#===========================
#Load The Model
#===========================

with open('medical_cost_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

#=========================
# The Logic Function
# ========================

def medical_cost_predict(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                            columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    #prediction
    prediction = model.predict(input_df)[0]   
    
    return f"Medical Insurance : {prediction:.2f}"

#======================================================
#App Interface
#======================================================
inputs = [
    gr.Number(label = 'Age'),
    gr.Radio(["male", "female"], label = 'Sex'),
    gr.Number(label = 'BMI'),
    gr.Slider(0, 7, step = 1, label = 'Children Number'),
    gr.Radio(['yes', 'no'], label = 'Smoker'),
    gr.Dropdown(['southwest', 'southeast', 'northwest', 'northeast'], label = 'Region')
]

medi_app = gr.Interface(
    fn = medical_cost_predict,
    inputs = inputs,
    outputs = 'text',
    title = 'Medical Insurance Cost Predictor'

)

#==============
#App Launch
#==============

medi_app.launch(share = True)