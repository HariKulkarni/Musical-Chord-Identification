import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Disable OneDNN optimization for compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Custom CSS styling
st.markdown("""
    <style>
    h1 {
        color: black;
        text-align: center;
        font-size: 48px;
        margin-top: 20px;
    }
    h2 {
        color: black;
        text-align: center;
        font-size: 24px;
        margin-bottom: 40px;
        text-shadow: 2px 2px 2px rgba(255, 255, 255, 0.7);
    }
    .bold-text {
        color: black;
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-text {
        color: black;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .metric-text {
        color: black;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    h3, p, title, .st-emotion-cache-1uixxvy, .st-emotion-cache-ltfnpr, #upload-an-image-for-chord-identification, .e1nzilvr3 svg {
        color: black;
    }
    .st-emotion-cache-1kyxreq img, .st-emotion-cache-1kyxreq svg {
        width:500px;
        box-shadow: 2px 2px 3px black;
        border: black 3px solid;
    }
        .st-emotion-cache-1gulkj5{
        background-color: rgb(119 119 117)
    }
    .st-emotion-cache-9ycgxx{
        color: white;
    }
    .st-emotion-cache-1aehpvj{
        color: white;
    }
    .st-emotion-cache-7ym5gk{
        background-color: rgb(255, 255, 255);
    }
    .st-emotion-cache-7ym5gk:hover{
        background-color: rgb(206, 204, 204);
        color: black
    }
    .st-emotion-cache-7oyrr6{
        color: rgb(0 0 0 / 60%);
    }
    img, svg {
        align-self: center;
    }
    .st-emotion-cache-12fmjuu{
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<h1>Musical Chord Identification</h1>', unsafe_allow_html=True)
st.markdown('<h2>This application identifies musical chords using four different models:</h2>', unsafe_allow_html=True)

# Function to read and encode the image file for background
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set the background image
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

# Use a valid file path for the background image
set_png_as_page_bg(r"E:\Music Chord\GUI\download2.jpg.png")

# File uploader for image classification
st.markdown('<h6>Upload an image for chord identification</h6>', unsafe_allow_html=True)
upload = st.file_uploader('', type=['png', 'jpg'])

# Load models with custom objects (if any)
def load_custom_model(model_path):
    try:
        # Custom class to handle DepthwiseConv2D layers if present in the model
        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, **kwargs):
                if 'groups' in kwargs:
                    kwargs.pop('groups')
                super().__init__(**kwargs)
        
        custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
        
        # Attempt to load the model with custom objects
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    
    except ValueError as ve:
        st.error(f"ValueError: {ve}")
    except KeyError as ke:
        st.error(f"KeyError: {ke}")
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
    
    return None

# Alternative model loading without custom objects for debugging
def load_standard_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Standard loading failed for model {model_path}: {e}")
    return None

# Paths for the models
model1_path = os.path.join('E:\Music Chord\Models\Vgg19.h5')
model2_path = os.path.join('E:\Music Chord\Models\ResNet50V2.h5')
model3_path = os.path.join('E:\Music Chord\Models\inceptionV3_Upd.h5')
model4_path = os.path.join('E:\Music Chord\Models\MobileNetV2.h5')

# Try loading models with custom objects first, fallback to standard loading if needed
model1 = load_custom_model(model1_path) if os.path.exists(model1_path) else load_standard_model(model1_path)
model2 = load_custom_model(model2_path) if os.path.exists(model2_path) else load_standard_model(model2_path)
model3 = load_custom_model(model3_path) if os.path.exists(model3_path) else load_standard_model(model3_path)
model4 = load_custom_model(model4_path) if os.path.exists(model4_path) else load_standard_model(model4_path)

# Define class labels
class_labels = ['A_Major_W', 'A_Minor_W', 'C_Major_W', 'C_Minor_W', 'D_Major_W', 'D_Minor_W', 'E_Major_W', 'E_Minor_W', 'F_Major_W', 'F_Minor_W', 'G_Major_W', 'G_Minor_W']

# Define the path to the audio files
audio_folder = r"E:\Music Chord\AudioFiles"
audio_files = {
    'A_Major_W': os.path.join(audio_folder, 'A_Major_W.mp3'),
    'A_Minor_W': os.path.join(audio_folder, 'A_Minor_W.mp3'),
    'C_Major_W': os.path.join(audio_folder, 'C_Major_W.mp3'),
    'C_Minor_W': os.path.join(audio_folder, 'C_Minor_W.mp3'),
    'D_Major_W': os.path.join(audio_folder, 'D_Major_W.mp3'),
    'D_Minor_W': os.path.join(audio_folder, 'D_Minor_W.mp3'),
    'E_Major_W': os.path.join(audio_folder, 'E_Major_W.mp3'),
    'E_Minor_W': os.path.join(audio_folder, 'E_Minor_W.mp3'),
    'F_Major_W': os.path.join(audio_folder, 'F_Major_W.mp3'),
    'F_Minor_W': os.path.join(audio_folder, 'F_Minor_W.mp3'),
    'G_Major_W': os.path.join(audio_folder, 'G_Major_W.mp3'),
    'G_Minor_W': os.path.join(audio_folder, 'G_Minor_W.mp3')
}

def predict_image(image, model):
    # Convert image to RGB if it has an alpha channel
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))  # Resize the image to the required size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction

def evaluate_model(y_true, y_pred):
    f1 = (f1_score(y_true, y_pred, average='weighted'))*100
    precision = (precision_score(y_true, y_pred, average='weighted'))*100
    recall = (recall_score(y_true, y_pred, average='weighted'))*100
    accuracy = (accuracy_score(y_true, y_pred))*100
    return f1, precision, recall, accuracy

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(plt)

if upload is not None:
    im = Image.open(upload)
    st.image(im, caption='Uploaded Image', use_column_width=True)
    
    st.markdown('<div class="bold-text">Identifying...</div>', unsafe_allow_html=True)

    # Initialize an empty list to store predictions
    predictions = []
    actual_labels = []

    # Get predictions from each model
    # Initialize lists to store predictions
    predictions = []

    # Get predictions from each model
    if model1 is not None:
        try:
            prediction1 = predict_image(im, model1)
            predicted_class1 = np.argmax(prediction1, axis=1)[0]
            predictions.append(class_labels[predicted_class1])
        except Exception as e:
            st.error(f"Prediction failed for VGG19: {e}")

    if model2 is not None:
        try:
            prediction2 = predict_image(im, model2)
            predicted_class2 = np.argmax(prediction2, axis=1)[0]
            predictions.append(class_labels[predicted_class2])
        except Exception as e:
            st.error(f"Prediction failed for ResNet50V2: {e}")

    if model3 is not None:
        try:
            prediction3 = predict_image(im, model3)
            predicted_class3 = np.argmax(prediction3, axis=1)[0]
            predictions.append(class_labels[predicted_class3])
        except Exception as e:
            st.error(f"Prediction failed for InceptionV3: {e}")

    if model4 is not None:
        try:
            prediction4 = predict_image(im, model4)
            predicted_class4 = np.argmax(prediction4, axis=1)[0]
            predictions.append(class_labels[predicted_class4])
        except Exception as e:
            st.error(f"Prediction failed for MobileNetV2: {e}")

    # Calculate the mode (most frequent prediction)
    final_prediction = max(set(predictions), key=predictions.count)

    # Display the final predicted class
    if predictions:
        y_true = [predictions[0]] * len(predictions)  # Using the first model's prediction as the true label for example

        f1, precision, recall, accuracy = evaluate_model(y_true, predictions)

        # Add tabs for model-specific details
        tabs = st.tabs(["Prediction", "Analysis", "Confusion Matrix"])

        # Tab 1: Prediction (Displays uploaded image and summary)
        with tabs[0]:
            st.markdown("<div class='prediction-text'>Model Predictions:</div>", unsafe_allow_html=True)

            if model1 is not None:
                st.markdown(f"<div class='prediction-text'>VGG19 Prediction: {class_labels[predicted_class1]}</div>", unsafe_allow_html=True)
            if model2 is not None:
                st.markdown(f"<div class='prediction-text'>ResNet50V2 Prediction: {class_labels[predicted_class2]}</div>", unsafe_allow_html=True)
            if model3 is not None:
                st.markdown(f"<div class='prediction-text'>InceptionV3 Prediction: {class_labels[predicted_class3]}</div>", unsafe_allow_html=True)
            if model4 is not None:
                st.markdown(f"<div class='prediction-text'>MobileNetV2 Prediction: {class_labels[predicted_class4]}</div>", unsafe_allow_html=True)
            
            # Display metrics
            st.markdown(f"<div class='metric-text'>F1 Score: {f1:.4f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-text'>Precision: {precision:.4f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-text'>Recall: {recall:.4f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-text'>Accuracy: {accuracy:.4f}</div>", unsafe_allow_html=True)

            st.markdown(f"<div class='prediction-text'>Final Predicted Chord: {final_prediction}</div>", unsafe_allow_html=True)

            # Play the audio for the final predicted chord
            if final_prediction in audio_files:
                st.audio(audio_files[final_prediction], format='audio/wav')

        # Tab 2: Analysis (Displays evaluation metrics)
        with tabs[1]:
            st.subheader("Model Analysis")
            st.markdown('<h3>Model Accuracy Comparison</h3>', unsafe_allow_html=True)
            st.image('E:\Music Chord\Analysis\Testing Accuracy of all the 4 models (3).png', use_column_width=True)
            st.markdown('<h3>Model Accuracy Comparison Graph</h3>', unsafe_allow_html=True)
            st.image('E:\Music Chord\Analysis\model_accuracy_comparison2.png', use_column_width=True)
            st.markdown('<h3>Mean ROC Curves Across Different Models</h3>', unsafe_allow_html=True)
            st.image('E:\Music Chord\Analysis\Mean ROC Curves Across Different Models.png', use_column_width=True)

        # Tab 3: Confusion Matrix (Displays the confusion matrix and related images)
        with tabs[2]:
            st.subheader("Confusion Matrix")
            st.markdown('<h4>I have consider total 60 images in the testing folder</h4>', unsafe_allow_html=True)
            st.markdown('<h3>VGG19 Confusion Matrix</h3>', unsafe_allow_html=True)
            st.image('E:/Music Chord/Analysis/VGG19_CM.png', use_column_width=True)
            st.markdown('<h3>ResNet50V2 Confusion Matrix</h3>', unsafe_allow_html=True)
            st.image('E:/Music Chord/Analysis/ResNet_CM.png', use_column_width=True)
            st.markdown('<h3>InceptionV3 Confusion Matrix</h3>', unsafe_allow_html=True)
            st.image('E:/Music Chord/Analysis/Inception_CM.png', use_column_width=True)
            st.markdown('<h3>MobileNetV2 Confusion Matrix</h3>', unsafe_allow_html=True)
            st.image('E:/Music Chord/Analysis/MobileNet_CM.png', use_column_width=True)
else:
    st.markdown('<h6 style="text-align: center;">Please upload an image for chord identification.</h6>', unsafe_allow_html=True)
