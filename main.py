import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import urllib
import json
import io

def load_model():
    # Load the pre-trained model resnet18
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    model.eval()
    return model

def load_labels():
    # Load the labels for the pre-trained model
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels_path, _ = urllib.request.urlretrieve(LABELS_URL)
    with open(labels_path) as f:
        labels = json.load(f)
    return labels

def preprocess_image(image):
    # Define the transformations for image preprocessing to fit the model resnet18
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

def classify_image(model, image):
    # Classify the image
    with torch.no_grad():
        output = model(image)
    
    # Get the predicted class label
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()

def main():
    st.title("Image Classification")
    st.write("Upload an image and let the model classify it!")
    
    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        
        # Load the model and labels
        model = load_model()
        labels = load_labels()
        
        # Preprocess and classify the image
        image = preprocess_image(image)
        predicted_idx = classify_image(model, image)
        predicted_label = labels[predicted_idx]
    
        # Display the predicted label
        st.write("Predicted Label:", predicted_label)

if __name__ == '__main__':
    main()
