import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define the transform for inference
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the model for cattle detection (Cow/Buffalo)
def load_cattle_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)  # 3 classes: Cow, Buffalo, None
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model for breed detection (Cow/Buffalo breeds)
def load_breed_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Class names for cattle detection (Cow, Buffalo, None)
cattle_class_names = ['Buffalo', 'Cow', 'None']
breed_names = ['Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss', 'Dangi', 
               'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 
               'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 
               'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 
               'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 
               'Umblachery', 'Vechur']

# Function to predict cattle (Cow or Buffalo)
def predict_cattle(image, model):
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = cattle_class_names[predicted.item()]

    return predicted_class, confidence.item()

# Function to predict the breed of an image
def predict_breed(image, model, breed_names):
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = breed_names[predicted.item()]

    return predicted_class

# Streamlit app
st.title("Cattle and Breed Classifier")
st.write("Upload an image to classify it into Cattle (Cow or Buffalo) and the respective breed if detected.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert('RGB')

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', width=400)

    # Load the cattle detection model
    cattle_model_path = 'models/best_cow_buffalo_none_classifier.pth'  # Path to cattle detection model
    cattle_model = load_cattle_model(cattle_model_path)

    with col2:
        # Detect cattle (Cow or Buffalo)
        with st.spinner('Classifying cattle...'):
            predicted_cattle, cattle_confidence = predict_cattle(image, cattle_model)

        # Check if the confidence is >= 60%
        if cattle_confidence >= 0.60:
            if predicted_cattle in ['Cow', 'Buffalo']:
                # Display the result of cattle classification
                st.success(f"##### Predicted Cattle: {predicted_cattle}")
                st.write(f"###### Confidence: {cattle_confidence * 100:.2f}%")
                # If the cattle is classified as Cow or Buffalo, show the Detect Breed button
                st.write("##### Now you can detect the breed!")

                # Add a button to detect the breed
                if st.button("Detect Breed"):
                    # Load the breed detection model based on the detected cattle
                    if predicted_cattle == 'Cow':
                        breed_model_path = 'models/breed_classifier.pth'  # Path to cow breed model
                        breed_model = load_breed_model(breed_model_path, len(breed_names))
                        breed_names = breed_names
                    else:
                        breed_model_path = 'models/breed_classifier.pth'  # Path to buffalo breed model
                        breed_model = load_breed_model(breed_model_path, len(breed_names))
                        breed_names = breed_names

                    # Predict the breed
                    with st.spinner(f"Classifying {predicted_cattle} breed..."):
                        predicted_breed = predict_breed(image, breed_model, breed_names)

                    # Display the breed result
                    st.success(f"##### Detected Breed: {predicted_breed}")
            else:
                st.warning("No cow or buffalo detected. Breed detection will not be performed.")
        else:
            # If confidence is below 60%, classify as None
            st.warning(f"Confidence is too low ({cattle_confidence * 100:.2f}%). Classifying as 'None'.")
            predicted_cattle = 'None'
