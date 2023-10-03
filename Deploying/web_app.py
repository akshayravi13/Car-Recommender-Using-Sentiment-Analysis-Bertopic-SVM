import numpy as np
import pickle
import streamlit as st

#loading saved model
loaded_model = pickle.load(open('D:/Sem VI/AOML/Deploying/trained_model.sav','rb'))

#creating a function for prediction
def brand_suggestion(input_data):

    prediction = loaded_model.predict(input_data)

    return prediction[0]

def main():

    #giving a title
    st.title('Car Recommender Web App')

    st.header("Select the features you want in your car")

    #getting the input data from the user
    price =  st.checkbox('Price')
    comfort =  st.checkbox('Comfort')
    drive =  st.checkbox('Drive')
    suv =  st.checkbox('SUV')
    sport =  st.checkbox('Sport')
    mpg =  st.checkbox('MPG')
    look =  st.checkbox('Look')
    coupe =  st.checkbox('Coupe')
    convertible =  st.checkbox('Convertible')
    interior =  st.checkbox('Interior')

    input_data = (price,comfort,drive,suv,sport,mpg,look,coupe,convertible,interior)
    input_data = np.asarray(input_data,dtype=int)
    input_data = input_data.reshape(1,-1)

    #code for prediction
    suggestion = ''

    #creating prediction button
    if st.button('Suggest Brand'):
        suggestion = brand_suggestion(input_data)

    st.success(suggestion)


if __name__=='__main__':
    main()
