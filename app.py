import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

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
    "burger": "🍔 **Calories**: 250-350 kcal per burger. High in protein and fat. For a healthier option, use lean meat, whole grain buns, and load up on veggies. 🥗",
    "Apple": "🍎 **Calories**: 52 kcal per 100g. Apples are a good source of dietary fiber and Vitamin C. Rich in antioxidants, they promote digestive health and are low in calories. Perfect for a healthy snack! 🥰",
    "Chapathi": "🥖 **Calories**: 120 kcal per piece (40g). Chapathi is made from whole wheat and provides complex carbohydrates, fiber, and some protein. It's a great energy booster and digestive aid! 💪",
    "Chicken gravy": "🍗 **Calories**: 250-300 kcal per 100g. Chicken gravy is protein-rich, great for muscle repair. Be mindful of fat content. Ideal for a hearty meal. 🍽️",
    "Fries": "🍟 **Calories**: 312 kcal per 100g. Fries are energy-packed but also high in unhealthy fats. Enjoy as a treat, not a daily dish! 🍔",
    "Idli": "🥞 **Calories**: 39 kcal per idli. Idlis are low in fat, packed with carbohydrates and protein. A light, easy-to-digest breakfast option for balanced energy. 🥱",
    "Pizza": "🍕 **Calories**: 285 kcal per slice. A combination of carbs (crust), fats (cheese), and protein (meat/cheese). Opt for veggies and whole grain crust for a healthier slice! 🌱",
    "Rice": "🍚 **Calories**: 130 kcal per 100g. Rice is a great source of energy, though low in protein. Pair with protein-rich foods for a complete meal. 🍛",
    "Soda": "🥤 **Calories**: 40-50 kcal per 100ml. Soda is high in sugars with little nutritional value. Limit its consumption to avoid empty calories. 🚫",
    "Tomato": "🍅 **Calories**: 18 kcal per 100g. Tomatoes are low in calories and high in Vitamin C and antioxidants. They promote heart health and fight inflammation! 💓",
    "Vada": "🍩 **Calories**: 190-250 kcal per vada. High in protein and carbohydrates, but also high in fats due to deep frying. Enjoy in moderation! ⚖️",
    "Banana": "🍌 **Calories**: 90 kcal per 100g. Bananas are rich in carbs and potassium, offering a great energy boost and aiding in muscle recovery. 💪🍃",
    "Burger": "🍔 **Calories**: 250-350 kcal per burger. High in protein and fat. For a healthier option, use lean meat, whole grain buns, and load up on veggies. 🥗"
}

# Function to process a single frame
def process_frame(img, detected_message_container):
    detected_messages = []  # List to collect custom messages for detected classes
    results = model.predict(img, verbose=False)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            class_index = int(box.cls[0])

            label = class_names[class_index] if class_index < len(class_names) else "Unknown"
            custom_message = custom_messages.get(label, f"Detected: {label}")
            
            # Collect message for stacking
            detected_messages.append(f"**Class:** {label}  \n**Message:** {custom_message}")

            # Draw bounding box and label
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if custom_message:
                cv2.putText(img, custom_message, (int(x1), int(y1) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display stacked custom messages if multiple classes are detected
    if detected_messages:
        detected_message_container.markdown("<br>".join(detected_messages), unsafe_allow_html=True)

    return img

# Streamlit app layout
st.title("Object Detection with YOLOv10")

# Option for input method
input_option = st.radio("Select Input Method", ["Upload Image", "Upload Video", "Webcam"], index=0)

# Placeholder for detected class and message
detected_message_container = st.empty()

# Add a stop button that will be used to control video playback
stop_video = False

if input_option == "Upload Image":
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Process uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image = process_frame(image, detected_message_container)
        st.image(processed_image, channels="BGR", caption="Processed Image", use_container_width=True)

elif input_option == "Upload Video":
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            video_path = temp_file.name
        
        # Open video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        # Add a stop button to allow stopping the video
        stop_video = st.button("Stop Video")
        if stop_video:
            cap.release()  # Stop the video playback

        while cap.isOpened() and not stop_video:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the current frame
            processed_frame = process_frame(frame, detected_message_container)

            # Convert frame to RGB for display in Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Display the processed frame
            stframe.image(processed_frame_rgb, channels="RGB", use_container_width=True)

        cap.release()  # Ensure video is released after completion or stop

elif input_option == "Webcam":
    st.warning("Ensure your webcam is connected and allowed.")
    if st.button("Start Webcam"):
        # Real-time video processing from webcam
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        # Add a stop button to stop the webcam feed
        stop_webcam = st.button("Stop Webcam")
        if stop_webcam:
            cap.release()  # Stop the webcam

        while cap.isOpened() and not stop_webcam:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the current frame
            processed_frame = process_frame(frame, detected_message_container)

            # Convert frame to RGB for display in Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Display the processed frame
            stframe.image(processed_frame_rgb, channels="RGB", use_container_width=True)

        cap.release()  # Ensure webcam is released after stopping
