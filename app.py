import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import gdown
import os
import torch.nn as nn

MODEL_PATH = "fairvision_deployed.pth"


if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1hAzoRxS9YxISqCjLnFp2wj6Aict7dh8-"
    gdown.download(url, MODEL_PATH, quiet=False)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(160 * 160 * 3, 2) 

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = MyModel()
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
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

    
    labels = ["Class 0", "Class 1"]  

    st.success(f"Prediction: {labels[prediction.item()]}")
