import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from yolov5.utils.datasets import letterbox

# Load the YOLOv7 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov7.pt')

# Define the Streamlit app
st.set_page_config(page_title='Object Detection', page_icon=':detective:', layout='wide')

st.title('Object Detection')

# Define the confidence threshold slider
confidence_threshold = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.1)

# Define the input image uploader
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define the output image placeholder
output_image = st.empty()

# Define the error handling for invalid images
try:
    if uploaded_file is not None:
        # Process the image with YOLOv7
        image = Image.open(uploaded_file)
        results = detect_objects(image)

        # Display the output image and additional information about the detected objects
        output_image.image(results.render(), use_column_width=True)
        for result in results.pandas().xyxy[0].iterrows():
            index, data = result
            class_name = data['class']
            confidence = data['confidence']
            output_image.caption(f'{class_name}: {confidence:.2f}')

except:
    st.error('Invalid file type or error processing image')

# Define the function to detect objects
@st.cache(suppress_st_warning=True, show_spinner=False)
def detect_objects(image):
    image = letterbox(image, new_shape=640)[0]
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    results = model(image, confidence=confidence_threshold)
    return results
