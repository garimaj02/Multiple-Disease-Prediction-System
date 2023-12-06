# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:45:46 2023

@author: garim
"""

import pickle
import lightgbm as lgb
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models
diabetes_model = pickle.load(open('C:/Users/garim/OneDrive/Desktop/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('C:/Users/garim/OneDrive/Desktop/heart_disease_model.sav','rb'))

model_filename = 'C:/Users/garim/OneDrive/Desktop/lgb_model_fold_0.txt'  # Adjust the filename accordingly
loaded_model = lgb.Booster(model_file=model_filename)

parkinsons_model = pickle.load(open('C:/Users/garim/OneDrive/Desktop/parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Dengue Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity', 'heart', 'person'],
                          default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Ancestors Diabetic value')
    with col2:
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diab_diagnosis = ''

    # Placeholder for Disclaimer and Prediction Result
    disclaimer_placeholder = st.empty()
    prediction_result_placeholder = st.empty()

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        # Validation checks
        try:
            Pregnancies = float(Pregnancies)
            Glucose = float(Glucose)
            BloodPressure = float(BloodPressure)
            SkinThickness = float(SkinThickness)
            Insulin = float(Insulin)
            BMI = float(BMI)
            DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
            Age = float(Age)
        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")
            st.stop()

        if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            st.warning("Please enter valid numeric values for all fields.")
        else:
            diab_prediction = diabetes_model.predict(
                [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'

            # Display Input Values
            st.subheader('Report:')
            st.write(f'- Number of Pregnancies: {Pregnancies}')
            st.write(f'- Glucose Level: {Glucose}')
            st.write(f'- Blood Pressure value: {BloodPressure}')
            st.write(f'- Skin Thickness value: {SkinThickness}')
            st.write(f'- Insulin Level: {Insulin}')
            st.write(f'- BMI value: {BMI}')
            st.write(f'- Diabetes Pedigree Function value: {DiabetesPedigreeFunction}')
            st.write(f'- Age of the Person: {Age}')

            # Display Prediction Result
            prediction_result_placeholder.subheader('Prediction Result:')
            prediction_result_placeholder.success(diab_diagnosis)

            # Update Disclaimer content dynamically
            disclaimer_content = """
                **Disclaimer:**
                This prediction is based on a machine learning model and should not be considered as a definitive diagnosis. Consult with a healthcare professional for accurate medical advice.
                """
            disclaimer_placeholder.markdown(disclaimer_content)


# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age', placeholder='Enter age between 1 and 100')
        
    with col2:
        sex = st.text_input('Sex (1 = male; 0 = female)', placeholder='Enter 0 or 1')
        if sex not in ['0', '1'] and sex != '':
           st.warning("Please enter either '0' or '1' for Sex.")
           
    with col3:
        cp = st.text_input('Chest Pain (0 >= 25%; 1 >= 50%; 2 >= 75%; 3 > 75%)',placeholder= 'Enter value between 0 and 3')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results (0 >= 25%; 1 >= 50%; 2 >= 75%)')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment (0 >= 25%; 1 >= 50%; 2 >= 75%)')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal (0 = normal; 1 = fixed defect; 2 = reversable defect)')
        
     # code for Prediction
    heart_diagnosis = ''
    
    # Placeholder for Disclaimer and Prediction Result
    disclaimer_placeholder = st.empty()
    prediction_result_placeholder = st.empty()
    
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
    # Validation checks
        try:
        # Convert input values to float
          age = float(age)
          sex = float(sex)
          cp = float(cp)
          trestbps = float(trestbps)
          chol = float(chol)
          fbs = float(fbs)
          restecg = float(restecg)
          thalach = float(thalach)
          exang = float(exang)
          oldpeak = float(oldpeak)
          slope = float(slope)
          ca = float(ca)
          thal = float(thal)
        except ValueError:
          st.warning("Please enter valid numeric values for all fields.")
          st.stop()

        if not all([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
           st.warning("Please enter valid numeric values for all fields.")
        else:    
           heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

           if (heart_prediction[0] == 1):
              heart_diagnosis = 'The person is having heart disease'
           else:
              heart_diagnosis = 'The person does not have any heart disease'
   
    # Display Input Values for Heart Disease Prediction
           st.subheader('Report:')
           st.write(f'- Age: {age}')
           st.write(f'- Sex: {sex}')
           st.write(f'- Chest Pain types: {cp}')
           st.write(f'- Resting Blood Pressure: {trestbps}')
           st.write(f'- Serum Cholestoral in mg/dl: {chol}')
           st.write(f'- Fasting Blood Sugar > 120 mg/dl: {fbs}')
           st.write(f'- Resting Electrocardiographic results: {restecg}')
           st.write(f'- Maximum Heart Rate achieved: {thalach}')
           st.write(f'- Exercise Induced Angina: {exang}')
           st.write(f'- ST depression induced by exercise: {oldpeak}')
           st.write(f'- Slope of the peak exercise ST segment: {slope}')
           st.write(f'- Major vessels colored by flourosopy: {ca}')
           st.write(f'- thal: 0 = normal; 1 = fixed defect; 2 = reversable defect: {thal}')

    
    # Display Prediction Result
           prediction_result_placeholder.subheader('Prediction Result:')
           prediction_result_placeholder.success(heart_diagnosis)
    
    # Update Disclaimer content dynamically
           disclaimer_content = """
            **Disclaimer:**
            This prediction is based on a machine learning model and should not be considered as a definitive diagnosis. Consult with a healthcare professional for accurate medical advice.
            """
           disclaimer_placeholder.markdown(disclaimer_content)
    
    
# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # Placeholder for Disclaimer and Prediction Result
    disclaimer_placeholder = st.empty()
    prediction_result_placeholder = st.empty()
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        # Validation checks
       try:
           fo = float(fo)
           fhi = float(fhi)
           flo = float(flo)
           Jitter_percent = float(Jitter_percent)
           Jitter_Abs = float(Jitter_Abs)
           RAP = float(RAP)
           PPQ = float(PPQ)
           DDP = float(DDP)
           Shimmer = float(Shimmer)
           Shimmer_dB = float(Shimmer_dB)
           APQ3 = float(APQ3)
           APQ5 = float(APQ5)
           APQ = float(APQ)
           DDA = float(DDA)
           NHR = float(NHR)
           HNR = float(HNR)
           RPDE = float(RPDE)
           DFA = float(DFA)
           spread1 = float(spread1)
           spread2 = float(spread2)
           D2 = float(D2)
           PPE = float(PPE)
       except ValueError:
           st.warning("Please enter valid numeric values for all fields.")
           st.stop()
       # Display Input Values
       st.subheader('Report:')
       st.write(f'- MDVP:Fo(Hz): {fo}')
       st.write(f'- MDVP:Fhi(Hz): {fhi}')
       st.write(f'- MDVP:Flo(Hz): {flo}')
       st.write(f'- MDVP:Jitter(%): {Jitter_percent}')
       st.write(f'- MDVP:Jitter(Abs): {Jitter_Abs}')
       st.write(f'- MDVP:RAP: {RAP}')
       st.write(f'- MDVP:PPQ: {PPQ}')
       st.write(f'- Jitter:DDP: {DDP}')
       st.write(f'- MDVP:Shimmer: {Shimmer}')
       st.write(f'- MDVP:Shimmer(dB): {Shimmer_dB}')
       st.write(f'- Shimmer:APQ3: {APQ3}')
       st.write(f'- Shimmer:APQ5: {APQ5}')
       st.write(f'- MDVP:APQ: {APQ}')
       st.write(f'- Shimmer:DDA: {DDA}')
       st.write(f'- NHR: {NHR}')
       st.write(f'- HNR: {HNR}')
       st.write(f'- RPDE: {RPDE}')
       st.write(f'- DFA: {DFA}')
       st.write(f'- spread1: {spread1}')
       st.write(f'- spread2: {spread2}')
       st.write(f'- D2: {D2}')
       st.write(f'- PPE: {PPE}')
        
       parkinsons_prediction = parkinsons_model.predict(
           [[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR,
             HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

       if parkinsons_prediction[0] == 1:
           parkinsons_diagnosis = "The person has Parkinson's disease"
       else:
           parkinsons_diagnosis = "The person does not have Parkinson's disease"
           
       # Display Prediction Result
       prediction_result_placeholder.subheader('Prediction Result:')
       prediction_result_placeholder.success(parkinsons_diagnosis)
 
        # Update Disclaimer content dynamically
       disclaimer_content = """
                **Disclaimer:**
                This prediction is based on a machine learning model and should not be considered as a definitive diagnosis. Consult with a healthcare professional for accurate medical advice.
                """
       disclaimer_placeholder.markdown(disclaimer_content)
       
# Dengue Prediction Page
if selected == 'Dengue Prediction':
    # page title
    st.title('Dengue Prediction using ML')

    # Getting the input data from the user
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        tempmax = st.text_input('Temprature Max')
    
    with col2:
        tempmin = st.text_input('Temprature Min')
            
    with col3:
        temp = st.text_input('Temprature')
        
    with col4:
        feelslikemax = st.text_input('Feels Like max')   
        
    with col5:
       feelslikemin = st.text_input('Feels Like min')     
        
    with col1:
        feelslike = st.text_input('Feels LIke')
        
    with col2:
        dew = st.text_input('Dew')
        
    with col3:
        humidity = st.text_input('Humidity')
        
    with col4:
        precip = st.text_input('Precipitation')
        
    with col5:
        precipprob = st.text_input('Precipitation Probability')
        
    with col1:
        precipcover = st.text_input('Precipitation Cover')
        
    with col2:
        snow = st.text_input('Snow')
        
    with col3:
        snowdepth = st.text_input('Snow depth')
        
    with col4:
         windspeed = st.text_input('Wind Speed')
         
    with col5:
        winddir = st.text_input('Wind direction')
        
    with col1:
        sealvelpressure = st.text_input('sealvel Pressure')
        
    with col2:
        cloudcover = st.text_input('Cloud Cover')
        
    with col3:
        visibility = st.text_input('Visibility')
        
    with col4:
        solarradiation = st.text_input('Solar Radiation')
        
    with col5:
        solarenergy = st.text_input('Solar Energy')
        
    with col1:
        uvindex = st.text_input('UV Index')
        
    with col2:    
        conditions = st.text_input('Conditions')
        
    with col3:
        stations = st.text_input('Stations')
        
    
    # Code for Dengue Prediction
    dengue_diagnosis = ''

    # Placeholder for Disclaimer and Prediction Result
    disclaimer_placeholder = st.empty()
    prediction_result_placeholder = st.empty()

    # Creating a button for Dengue Prediction
    if st.button('Dengue Test Result'):
        # Validation checks
        try:
            tempmax = float(tempmax)
            tempmin = float(tempmin)
            temp = float(temp)
            feelslikemax = float(feelslikemax)
            feelslikemin = float(feelslikemin)
            feelslike = float(feelslike)
            dew = float(dew)
            humidity = float(humidity)
            precip = float(precip)
            precipprob = float(precipprob)
            precipcover = float(precipcover)
            snow = float(snow)
            snowdepth = float(snowdepth)
            windspeed = float(windspeed)
            winddir = float(winddir)
            sealvelpressure = float(sealvelpressure)
            cloudcover = float(cloudcover)
            visibility = float(visibility)
            solarradiation = float(solarradiation)
            uvindex = float(uvindex)
            conditions = float(conditions)
            stations = float(stations)
          
        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")
            st.stop()

       

        # Display Input Values for Dengue Prediction
        st.subheader('Report:')
        # Add lines to display input values for Dengue prediction
        
        dengue_prediction = loaded_model.predict(
            [[tempmax,tempmin,temp,feelslikemax,feelslikemin,feelslike,dew,humidity,precip,precipprob,precipcover,snow,snowdepth,windspeed,winddir,sealvelpressure,cloudcover,visibility,solarradiation,solarenergy,uvindex,conditions,stations]])

        
        # Display Dengue Prediction Result
        prediction_result_placeholder.subheader('Prediction Result:')
        if dengue_prediction[0] == 1:
            prediction_result_placeholder.success('Dengue Positive')
        else:
            prediction_result_placeholder.success('Dengue negative')

        # Update Disclaimer content dynamically
        disclaimer_content = """
            **Disclaimer:**
            This prediction is based on a machine learning model and should not be considered as a definitive diagnosis. Consult with a healthcare professional for accurate medical advice.
            """
        disclaimer_placeholder.markdown(disclaimer_content)
       
       