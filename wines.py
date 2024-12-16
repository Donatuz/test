import streamlit as st
import pickle
import numpy as np
import pandas as pd

#Loading the pickel files
loaded_model = pickle.load(open('models/wine_model.sav', 'rb'))

def wine_prediction(input_data):
    input_data_as_numpy = np.asarray(input_data)

    input_data_reshape = input_data_as_numpy.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshape)

    if prediction[0]==1:
        return "Good quality wine."
    else:
        return 'Bad quality wine' 


#Streamlit styling
def main():
    st.title('Wine Quality Prediction')
    
    #Collecting user input
    col1, col2, col3 =  st.columns(3)
    with col1:
        fixed_acidity = st.number_input('Fixed Acidity', value=None)
    
    with col2:
        volatile_acidity = st.number_input('Volatile Acidity', value=None)
    
    with col3:
        citric_acid = st.number_input('Citric Acid', value=None)
    

    with col1:
        residual_sugar = st.number_input('Residual Sugar', value=None)
    
    with col2:
        chlorides = st.number_input('chlorides', value=None)
    
    with col3:
        free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', value=None)
    
    with col1:
        total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', value=None)
    
    with col2:
        density = st.number_input('Density', value=None)
    
    with col3:
        pH = st.number_input('PH', value=None)
    
    with col1:
        sulphates = st.number_input('Sulphates', value=None)
    
    with col2:
        alcohol = st.number_input('Alcohol', value=None)
    
    if st.button('Wine Quality'):
        try:
            input_data = [
                float(fixed_acidity),
                float(volatile_acidity),
                float(citric_acid),
                float(residual_sugar),
                float(chlorides),
                float(free_sulfur_dioxide),
                float(total_sulfur_dioxide),
                float(density),
                float(pH),
                float(sulphates),
                float(alcohol)
            ]
            
            wine = wine_prediction(input_data)
            
            st.success(wine)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            
            
# Run the app
if __name__ == "__main__":
    main()