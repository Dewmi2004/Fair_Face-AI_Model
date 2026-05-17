import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import gdown
import os

MODEL_PATH = "fairvision_deployed.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/drive/folders/1Tsbg-K9YnxYM5tAUi2dIAF9b2fAsyLKK?usp=sharing"
    gdown.download(url, MODEL_PATH, quiet=False)

model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

st.title("FairFace AI Model 👤")
st.write("Upload an image to predict")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output, dim=1)

    st.success(f"Prediction: {prediction.item()}")
