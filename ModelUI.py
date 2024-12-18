import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from streamlit_option_menu import option_menu

# Set up the page configuration
st.set_page_config(page_title="Smart Shroom", layout="wide", page_icon="https://i.ibb.co/Gk3pM1t/Group-21.png")
# Function to apply gradient background
def set_background():
    gradient_css = """
    <style>
    /* Remove Streamlit's default header */
    header {visibility: hidden;}
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to top, #cdb4db, #ffc8dd, #ffafcc, #bde0fe);
        background-size: cover;
    }
    </style>
    """
    st.markdown(gradient_css, unsafe_allow_html=True)
    st.markdown(
        """
        <style>
            html, body {
                margin: 0;
                padding: 0;
                overflow-x: hidden; /* Prevent horizontal scrolling */
                width: 100%;
            }
            header {visibility: hidden;} /* Hide Streamlit's default header */

            /* Header container: stays fixed on top */
            .header-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 70px;
                background-color: #6ab17e;
                z-index: 2000; /* Header sits on top */
                display: flex;
                align-items: center;
                color: white;
                padding: 0 20px;
                box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
            }

            .title {
                font-size: 32px;
                font-weight: bold;
                margin-left: 10px;
                font-family: Times New Roman, serif;
            }

            .title-logo img {
                height: 35px;
            }

            /* Sidebar: ensure it stays below the header */
            [data-testid="stSidebar"] {
                position: fixed;
                left: 0;
                width: 200px;  /* Fixed width */
                height: 100%;
                z-index: 1000; /* Sidebar below header */
            }

            /* Adjust sidebar items if needed */
            [data-testid="stSidebarNav"] {
                top: 70px; /* Adjust to match header height */
            }

            /* Hide the sidebar expand/collapse arrow */
            [data-testid="stSidebarNav"] > button {
                display: none; /* Hides the button with the arrow */
            }
        </style>
        <div class="header-container">
            <div class="title-logo">
                <img src="https://i.ibb.co/Gk3pM1t/Group-21.png">
            </div>
            <div class="title">Smart Shroom</div>
        </div>
        """,
        unsafe_allow_html=True
    )
# Apply CSS to prevent the sidebar from collapsing
# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model2.keras")  # Replace with your model path
    return model


# Function to preprocess the image for classification
def preprocess_image(img):
    img = img.resize((224, 224))  # Assuming your model requires 224x224 input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Function to classify the mushroom image
def classify_mushroom(img, model):
    preprocessed_image = preprocess_image(img)
    prediction = model.predict(preprocessed_image)[0][0]  # Assuming binary classification
    return prediction


# Home page
def show_home_page():
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                background-image: url('https://i.ibb.co/yNnJPfv/imageedit-1-2473951278.jpg'); /* Background image */
                background-size: cover; /* Ensure the image covers the entire container */
                background-position: center; /* Center the image */
                background-attachment: fixed; /* Ensure the background stays in place */
            }
            .text-box {
                background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent white for better text visibility */
                padding: 20px; 
                border-radius: 10px; /* Rounded corners */
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5); /* Add a shadow for depth */
                margin: 50px auto; /* Center the box vertically and horizontally */
                max-width: 800px; /* Limit width for readability */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Welcome message with a styled container
    st.markdown(
        """
        <div class="text-box">
            <h4 style="text-align: center;"><b><i>Mushroom Identification Made Simple: Poisonous or Safe?</i></b></h4>
            <p style="text-align: center;">Welcome to <b>Smart Shroom</b>, your reliable tool for mushroom identification.</p>
            <p>This application is designed to assist in determining whether the mushroom in your image is the poisonous <b><i>Lepiota cristata</i></b>
            or the edible and safe <b><i>Coprinus comatus</i></b>.</p>
            <p>Simply upload a clear picture, and let our intelligent prediction system guide you toward safer decisions in identifying wild mushrooms.</p>
            <p style="text-align: center;"><b>—Stay curious, stay safe! 🍄</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Predict page
def show_predict_page():
    st.title("Classify your mushroom using an image.")

    # Option to choose the image input method
    input_method = st.radio("Select an input method:", ("Upload Image", "Use Camera"))

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image of a mushroom", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            process_and_predict(img, "Uploaded Mushroom")

    elif input_method == "Use Camera":
        picture = st.camera_input("Take a picture of the mushroom")
        if picture is not None:
            img = Image.open(picture)
            process_and_predict(img, "Captured Mushroom")

def process_and_predict(img, source):
    """Handles displaying the image and making predictions."""
    # Load the model (replace with your model's path)
    model = load_model()

    # Display the image
    st.image(img, caption=source)

    # Predict button
    if st.button("Predict"):
        st.write("Classifying the mushroom...")
        confidence = classify_mushroom(img, model)
        if confidence > 0.5:
            st.write(f"Prediction : **Not-Edible**")
            st.write(f"Confidence: **{confidence * 100:.2f}%**")
        else:
            st.write(f"Prediction : **Edible**")
            st.write(f"Confidence: **{(1 - confidence) * 100:.2f}%**")

# About page
def show_about_page():
    st.markdown(
        """
        <div>
            <h1>About</h1>
            <h3>The App</h3>
            <p>
                Smart Shroom leverages the power of deep learning, using models like <b>EfficientNetV2S</b>, to accurately 
                identify mushrooms as either edible or poisonous, which focuses on Lepiota cristata and Coprinus comatus. 
                Built with <b>Streamlit</b>, this web application allows 
                users to upload images of mushrooms for predictions. With Smart Shroom, you can make safer 
                decisions and ensure peace of mind when identifying wild mushrooms.
            </p>
            <h3>The Dataset</h3>
            <p>
                The dataset used to train the model was sourced from <i><b>Kaggle</i></b> and <i><b>iNaturalist</i></b>. These 
                platforms provided a focused collection of mushroom images, enabling the model to accurately differentiate 
                between two specific species: Lepiota cristata and Coprinus comatus.</p>
            <p>
                Source: <br><a href="https://www.kaggle.com/datasets/zedsden/mushroom-classification-dataset" target="_blank">
                        Kaggle - Datasets</a><br>
                        <a href="https://www.inaturalist.org/taxa/58694-Lepiota-cristata" target="_blank">
                        iNaturalist - Datasets</a>
            </p>
            <h3>The Developers</h3>
            <p>This Streamlit application was developed as a Learning Evidence for the subject CSDS 314 - Machine Learning and CS 3110 - Modelling and Simulation.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image("mejorada.jpg", width=200)  # Local image
        st.markdown(
            """
            **James Ian R. Mejorada**  
            **Email:** [jmejorada211@gmail.com](mailto:jmejorada211@gmail.com)  
            **Facebook:** [James Ian Mejorada](https://www.facebook.com/jamesian.mejorada)
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.image("dorin.jpg", width=200)  # Local image
        st.markdown(
            """
            **Kyla Bea C. Dorin**  
            **Email:** [kylabeadorin@gmail.com](mailto:kylabeadorin@gmail.com)  
            **Facebook:** [Kyla Bea Dorin](https://www.facebook.com/kyla.bea.dorin.2024)
            """,
            unsafe_allow_html=True
        )


# Apply gradient background
set_background()

# Sidebar navigation
with st.sidebar:
    
    menu = option_menu("", ["Home", "Predict", "About"],
        icons=["house", "camera", "info-circle"],
                       menu_icon="cast",
                       default_index=0,
                       styles={"nav-link-selected": {"background-color": "#4DA674", "color": "white"}},
                       )

# Show pages based on sidebar selection
if menu == "Home":
    show_home_page()
elif menu == "Predict":
    show_predict_page()
elif menu == "About":
    show_about_page()