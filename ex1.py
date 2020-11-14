import streamlit as st
import pandas as pd
import pickle 

st.write(""" 

## My First Web Application 
Let's enjoy **data science** project! 

""")

st.sidebar.header('User Input') 
st.sidebar.subheader('Please enter your data:')

# -- Define function to display widgets and store data
def get_input():
    # Display widgets and store their values in variables
    v_Sex = st.sidebar.radio('Sex', ['Male','Female','Infant'])
    v_Length = st.sidebar.slider('Length', 0.075, 0.745, 0.5)
    v_Diameter = st.sidebar.slider('Diameter', 0.05, 0.6, 0.4)
    v_Height = st.sidebar.slider('Height', 0.01, 0.24, 0.13)
    v_Whole_weight = st.sidebar.slider('Whole Weight', 0.002, 2.55, 0.78)
    v_Shucked_weight = st.sidebar.slider('Shucked Weight', 0.001, 1.07 , 0.3)
    v_Viscera_weight = st.sidebar.slider('Viscera Weight', 0.0005, 0.54, 0.17)
    v_Shell_weight = st.sidebar.slider('Shell Weight', 0.0015, 1.0, 0.24)

    # Change the value of sex to be {'M', 'F', 'I'} as stored in the trained dataset
    if v_Sex == 'Male':
        v_Sex = 'M'
    elif v_Sex == 'Female':
        v_Sex = 'F'
    else:
        v_Sex = 'I'

    # Store user input data in a dictionary
    data = {'Sex': v_Sex,
            'Length': v_Length,
            'Diameter': v_Diameter,
            'Height': v_Height,
            'Whole_weight': v_Whole_weight,
            'Shucked_weight': v_Shucked_weight,
            'Viscera_weight': v_Viscera_weight,
            'Shell_weight': v_Shell_weight}

    # Create a data frame from the above dictionary
    data_df = pd.DataFrame(data, index=[0])
    return data_df

# -- Call function to display widgets and get data from user
df = get_input()

st.header('Application of Abalone\'s Age Prediction:')

# -- Display new data from user inputs:
st.subheader('User Input:')
st.write(df)

# -- Data Pre-processing for New Data:
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample],axis=0)

#One-hot encoding for nominal features
cat_data = pd.get_dummies(df[['Sex']])

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)

#Drop un-used feature
X_new = X_new.drop(columns=['Sex'])

# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(X_new)

# -- Reads the saved normalization model
load_sc = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_sc.transform(X_new)

# -- Display normalized new data:
st.subheader('Normalized Input:')
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)

# -- Display predicted class:
st.subheader('Prediction:')
#penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(prediction)    