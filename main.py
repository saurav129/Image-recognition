import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import urllib
import json
import io

# Load the pre-trained model resnet18
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.eval()

# Define the transformations for image preprocessing to fit the model resnet18
#transform.Resize(256): resize the image to 256 pixels
#transform.CenterCrop(224): Crop to 224x224 pixels
#transforms.ToTensor(): Convert the datatype of the image to pytorch tensor
#transform.Normalize(): Generalizing the variations from 0 for each pixel, to make it fit the maodel
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the labels for the pre-trained model
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels_path, _ = urllib.request.urlretrieve(LABELS_URL)
with open(labels_path) as f:
    labels = json.load(f)

st.title("Image Classification")
st.write("Upload an image and let the model classify it!")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Check if a file was uploaded
if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)
    
    # Preprocess the image
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    # Classify the image
    with torch.no_grad():
        output = model(image)

    # Get the predicted class label
    _, predicted_idx = torch.max(output, 1)
    predicted_label = labels[predicted_idx.item()]


    # Display the predicted label
    st.write("Predicted Label:", predicted_label)
