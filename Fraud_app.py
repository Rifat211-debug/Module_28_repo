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

with open('fraud_detection_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

#=========================
# The Logic Function
# ========================

def detect_fraud(ClaimAmount, PatientAge, PatientGender,
       ProviderSpecialty, ClaimStatus, PatientIncome,
       PatientMaritalStatus, PatientEmploymentStatus, ProviderLocation,
       ClaimType, ClaimSubmissionMethod):
    input_df = pd.DataFrame([[ClaimAmount, PatientAge, PatientGender,
       ProviderSpecialty, ClaimStatus, PatientIncome,
       PatientMaritalStatus, PatientEmploymentStatus, ProviderLocation,
       ClaimType, ClaimSubmissionMethod]],
       columns = ['ClaimAmount', 'PatientAge', 'PatientGender', 'ProviderSpecialty',
       'ClaimStatus', 'PatientIncome', 'PatientMaritalStatus',
       'PatientEmploymentStatus', 'ProviderLocation', 'ClaimType',
       'ClaimSubmissionMethod'])
    #Prediction
    prediction = model.predict(input_df)[0]

    label_map = {
        0 : 'Legitimate',
        1 : 'Fraud'
    }

    return f'The Applicant is : {label_map[prediction]}'

#======================================================
#App Interface
#Defining inputs in a list to keep it clean
#======================================================

inputs = [
    gr.Number(label = 'Claim Amount'),
    gr.Number(label = 'Patient Age'),
    gr.Radio(['M', 'F'], label = 'Patient Sex'),
    gr.Dropdown (['Orthopedics', 'Cardiology', 'Neurology', 'Pediatrics',
       'General Practice'], label = 'Provider Speciality'),
    gr.Radio(['Pending', 'Denied', 'Approved'], label = 'Claim Status'),
    gr.Number(label = 'Patient Income'),
    gr.Dropdown(['Single', 'Widowed', 'Married', 'Divorced'], label = 'Patient Maritial Status'),
    gr.Dropdown(['Employed', 'Student', 'Unemployed', 'Retired'], label = 'Paptient Employment Status'),
    gr.Textbox(label = 'Patient Location'),
    gr.Dropdown(['Inpatient', 'Emergency', 'Routine', 'Outpatient'], label = 'Claim Type'),
    gr.Dropdown(['Paper', 'Online', 'Phone'], label = 'Claim Submission Method')
]

fraud_app = gr.Interface(
    fn = detect_fraud,
    inputs = inputs,
    outputs = 'text',
    title = 'Health Insurance Claim Approval Status'
)

#==============
#App Launch
#==============

fraud_app.launch(share = True)