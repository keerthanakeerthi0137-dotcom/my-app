import streamlit as st

st.title("My First Streamlit App 🎉")
st.write("Hello, welcome to my web app!")

number = st.slider("Pick a number", 1, 10, 5)
st.write("You selected:", number)
