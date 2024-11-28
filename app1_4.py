import subprocess
import sys

# Function to ensure required libraries are installed
def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        print(f"An error occurred while installing requirements: {e}")

install_requirements()

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time  # For adding delay to messages only


# Load the YOLO model
model = YOLO('food.pt')

# Define class names
class_names = ['Apple', 'Chapathi', 'Chicken Gravy', 'Fries', 'Idli', 'Pizza', 'Rice', 'Soda', 'Tomato', 'Vada', 'banana', 'burger']

# Define custom messages with macronutrient info and total calories, now with emotes
custom_messages = {
    "apple": "🍎 **Calories**: 52 kcal per 100g. Apples are a good source of dietary fiber and Vitamin C. Rich in antioxidants, they promote digestive health and are low in calories. Perfect for a healthy snack! 🥰",
    "chapathi": "🥖 **Calories**: 120 kcal per piece (40g). Chapathi is made from whole wheat and provides complex carbohydrates, fiber, and some protein. It's a great energy booster and digestive aid! 💪",
    "chicken gravy": "🍗 **Calories**: 250-300 kcal per 100g. Chicken gravy is protein-rich, great for muscle repair. Be mindful of fat content. Ideal for a hearty meal. 🍽️",
    "fries": "🍟 **Calories**: 312 kcal per 100g. Fries are energy-packed but also high in unhealthy fats. Enjoy as a treat, not a daily dish! 🍔",
    "idli": "🥞 **Calories**: 39 kcal per idli. Idlis are low in fat, packed with carbohydrates and protein. A light, easy-to-digest breakfast option for balanced energy. 🥱",
    "pizza": "🍕 **Calories**: 285 kcal per slice. A combination of carbs (crust), fats (cheese), and protein (meat/cheese). Opt for veggies and whole grain crust for a healthier slice! 🌱",
    "rice": "🍚 **Calories**: 130 kcal per 100g. Rice is a great source of energy, though low in protein. Pair with protein-rich foods for a complete meal. 🍛",
    "soda": "🥤 **Calories**: 40-50 kcal per 100ml. Soda is high in sugars with little nutritional value. Limit its consumption to avoid empty calories. 🚫",
    "tomato": "🍅 **Calories**: 18 kcal per 100g. Tomatoes are low in calories and high in Vitamin C and antioxidants. They promote heart health and fight inflammation! 💓",
    "vada": "🍩 **Calories**: 190-250 kcal per vada. High in protein and carbohydrates, but also high in fats due to deep frying. Enjoy in moderation! ⚖️",
    "banana": "🍌 **Calories**: 90 kcal per 100g. Bananas are rich in carbs and potassium, offering a great energy boost and aiding in muscle recovery. 💪🍃",
    "burger": "🍔 **Calories**: 250-350 kcal per burger. High in protein and fat. For a healthier option, use lean meat, whole grain buns, and load up on veggies. 🥗"
}

# Function to process a single frame with real-time detection and delayed message display
def process_frame_with_real_time_detection(img, detected_message_container, detected_classes, last_update_time, delay=1.5):
    # Perform detection
    results = model.predict(img, verbose=False)
    message_list = []  # To store all detected class messages
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            class_index = int(box.cls[0])

            label = class_names[class_index] if class_index < len(class_names) else "Unknown"
            custom_message = custom_messages.get(label.lower(), f"Detected: {label}")

            # Draw bounding box and label in real-time
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Add the message if it's a new detection (preventing repeated detections for the same frame)
            if custom_message and label not in detected_classes:
                detected_classes.append(label)
                message_list.append(custom_message)

    # Only update the detected message box after the specified delay
    current_time = time.time()
    if current_time - last_update_time > delay:
        # Update the message box with new messages
        if message_list:
            detected_message_container.markdown("### 🍽️ Detected Items 🍽️")
            for message in message_list:
                detected_message_container.markdown(f"{message}  \n")
            last_update_time = current_time  # Update the last update time

    return img, last_update_time

# Streamlit app layout
st.title("🍴 **Food Detection with YOLOv10** 🍴")
st.markdown("Welcome to the **Food Detection App**! Upload an image, video, or use your webcam to see the food detection with detailed macronutrient information. 🧑‍🍳")

# Option for input method
input_option = st.radio("📸 **Select Input Method**", ["Upload Image", "Upload Video", "Webcam"], index=0)

# Placeholder for detected class and message
detected_message_container = st.empty()
detected_classes = []  # List to store detected class names for message delay
last_update_time = time.time()  # Store the last update time for the custom message

if input_option == "Upload Image":
    uploaded_image = st.file_uploader("📷 **Choose an image file**", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Process uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image, last_update_time = process_frame_with_real_time_detection(image, detected_message_container, detected_classes, last_update_time)
        st.image(processed_image, channels="BGR", caption="Processed Image", use_container_width=True)

elif input_option == "Upload Video":
    uploaded_video = st.file_uploader("🎥 **Choose a video file**", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            video_path = temp_file.name

        # Real-time video processing
        stframe = st.empty()
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the current frame with real-time detection and delayed message
            processed_frame, last_update_time = process_frame_with_real_time_detection(frame, detected_message_container, detected_classes, last_update_time)

            # Display the processed frame
            stframe.image(processed_frame, channels="BGR", use_container_width=True)

        cap.release()

elif input_option == "Webcam":
    st.warning("🚨 Ensure your webcam is connected and allowed. 🚨")
    if st.button("🎥 **Start Webcam**"):
        # Real-time video processing from webcam
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the current frame with real-time detection and delayed message
            processed_frame, last_update_time = process_frame_with_real_time_detection(frame, detected_message_container, detected_classes, last_update_time)

            # Display the processed frame
            stframe.image(processed_frame, channels="BGR", use_container_width=True)

        cap.release()
