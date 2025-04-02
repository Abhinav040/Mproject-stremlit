import streamlit as st
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import requests  # For chatbot API call

# Load model and class indices
model_path = "./trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open("class_indices.json"))

# Function to preprocess image
def load_and_preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Normalize
    return img_array

# Function to predict disease
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to query Python chatbot API
def query_chatbot(question):
    url = "https://devbots-server.vercel.app/api/chatbots/chat"  # Replace with your Python chatbot API URL
    payload = {
        "apiKey": "c77089c2595d3851d59a2afa2a75be1363267c24e36dfc69b43e645dc4f819e3",  # Replace with your actual API key
        "query": question
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json().get("response", "No response found.")
    else:
        return f"Error: {response.status_code}, {response.text}"

# --- Streamlit UI ---

st.set_page_config(
    page_title="Plant Disease Classifier 🌿",
    page_icon="🌱",
    layout="wide"
)

st.title("🌿 Plant Disease Classifier")
st.write("Upload an image to detect plant diseases & get solutions!")

# File uploader
uploaded_image = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Create two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Preview 📷")
        resized_img = image.resize((200, 200))
        st.image(resized_img)

    with col2:
        st.markdown("### Classification 🧪")
        if st.button('🔍 Classify Image'):
            with st.spinner("Analyzing Image... ⏳"):
                prediction = predict_image_class(model, image, class_indices)
                # Display classification result
                st.success(f"🌿 Prediction: **{prediction}**")
                # Check if the prediction contains the word "healthy"
                if "healthy" in prediction.lower():
                    st.markdown("🎉 Your plant looks healthy! Keep up the good care!")
                else:
                    # Auto-query chatbot with disease name
                    chatbot_response = query_chatbot(f"How to fix {prediction}?")
                    st.markdown(f"**🤖 Chatbot Advice:** {chatbot_response}")

# --- Chatbot UI ---
st.markdown("## 💬 LeafCure Chatbot ")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
# User input for chatbot

user_input = st.chat_input("Ask me about plant diseases...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

     # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Get chatbot response
    bot_response = query_chatbot(user_input)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Display chatbot response
    with st.chat_message("assistant"):
        st.write(bot_response)

