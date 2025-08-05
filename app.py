import gradio as gr
import joblib
import pandas as pd

# Load model and encoders
artifacts = joblib.load("travel_dest_predictor.pkl")
model = artifacts['model']
le_name = artifacts['le_name']
le_country = artifacts['le_country']

def predict(season, travel_month, budget, trip_duration_days, group_size, companion_type, interest):
    df_input = pd.DataFrame([{
        'season': season,
        'travel_month': travel_month,
        'budget': float(budget),
        'trip_duration_days': int(trip_duration_days),
        'group_size': int(group_size),
        'companion_type': companion_type,
        'interest': interest
    }])
    prediction = model.predict(df_input)[0]
    dest_name = le_name.inverse_transform([prediction[0]])[0]
    dest_country = le_country.inverse_transform([prediction[1]])[0]
    return dest_name, dest_country

# Dropdown values (replace with your actual data if needed)
seasons = ['Winter', 'Summer', 'Spring', 'Autumn']
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']
companions = ['Solo', 'Family', 'Friends', 'Partner']
interests = ['Adventure', 'Culture', 'Relaxation', 'Nature']

inputs = [
    gr.Dropdown(seasons, label="Season"),
    gr.Dropdown(months, label="Travel Month"),
    gr.Number(label="Budget"),
    gr.Number(label="Trip Duration (days)"),
    gr.Number(label="Group Size"),
    gr.Dropdown(companions, label="Companion Type"),
    gr.Dropdown(interests, label="Interest")
]

outputs = [
    gr.Textbox(label="Predicted Destination Name"),
    gr.Textbox(label="Predicted Destination Country")
]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="Travel Destination Recommender",
    description="Enter your preferences to get a recommended travel destination."
)

demo.launch()
