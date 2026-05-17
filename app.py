"""
FairVision Streamlit App — Age Classification Demo
Model: FairVisionCNN_v3 (trained on FairFace 0.25)
Run with: streamlit run fairvision_app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import io
import os


st.set_page_config(
    page_title="FairVision · Age Classifier",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e8e8f0;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}

.hero-sub {
    text-align: center;
    color: #9ca3af;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

.result-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    backdrop-filter: blur(8px);
}

.prediction-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #a78bfa;
    text-align: center;
}

.confidence-pct {
    font-size: 1.1rem;
    color: #60a5fa;
    text-align: center;
    font-weight: 500;
    margin-top: 0.2rem;
}

.pill {
    display: inline-block;
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    color: #34d399;
    font-weight: 500;
    margin: 0.15rem;
}

.info-box {
    background: rgba(96,165,250,0.08);
    border-left: 3px solid #60a5fa;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #cbd5e1;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.8rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(124,58,237,0.4);
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #60a5fa);
    border-radius: 999px;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Sequential()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.se(self.conv(x)) + self.shortcut(x))


class FairVisionCNN(nn.Module):
    def __init__(self, num_classes=9, drop=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.stage1 = nn.Sequential(ResBlock(32, 64), ResBlock(64, 64))
        self.stage2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.stage3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.stage4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(drop / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str, meta_path: str):
    with open(meta_path) as f:
        meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FairVisionCNN(num_classes=meta["num_classes"]).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, meta, device



def build_tta_transforms(img_size: int, mean, std):
    def norm():
        return transforms.Normalize(mean, std)

    return [
        transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(), norm()]),
        transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), norm()]),
        transforms.Compose([transforms.Resize((img_size + 12, img_size + 12)),
                             transforms.CenterCrop(img_size),
                             transforms.ToTensor(), norm()]),
        transforms.Compose([transforms.Resize((img_size + 12, img_size + 12)),
                             transforms.CenterCrop(img_size),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), norm()]),
        transforms.Compose([transforms.Resize((img_size + 8, img_size + 8)),
                             transforms.CenterCrop(img_size),
                             transforms.ToTensor(), norm()]),
    ]


@torch.no_grad()
def predict(model, image: Image.Image, meta: dict, device):
    tfms = build_tta_transforms(meta["img_size"], meta["mean"], meta["std"])
    probs_sum = None
    for tfm in tfms:
        tensor = tfm(image).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        probs_sum = probs if probs_sum is None else probs_sum + probs
    avg_probs = probs_sum / len(tfms)
    pred_idx = int(np.argmax(avg_probs))
    return pred_idx, avg_probs


with st.sidebar:
    st.markdown("### ⚙️ Model Files")
    weights_path = st.text_input(
        "Weights (.pth)",
        value="fairvision_deployed.pth",
        help="Path to fairvision_deployed.pth",
    )
    meta_path = st.text_input(
        "Metadata (.json)",
        value="model_meta.json",
        help="Path to model_meta.json",
    )
    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "FairVisionCNN_v3 is trained on FairFace 0.25 with fairness-aware "
        "techniques: Balanced Subset, Focal Loss, Soft Class Weights, "
        "WeightedRandomSampler, and Test-Time Augmentation (TTA × 5)."
    )


st.markdown('<div class="hero-title">👁️ FairVision</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Fairness-aware age classification · FairVisionCNN v3</div>',
    unsafe_allow_html=True,
)

if not os.path.exists(weights_path) or not os.path.exists(meta_path):
    st.error(
        f"Model files not found. Place **{weights_path}** and **{meta_path}** "
        "in the same directory as this script, or update the paths in the sidebar."
    )
    st.stop()

with st.spinner("Loading model weights…"):
    model, meta, device = load_model(weights_path, meta_path)

cols = st.columns(3)
cols[0].metric("Baseline Acc", f"{meta['baseline_acc']*100:.1f}%")
cols[1].metric("Mit-1 Acc", f"{meta['mit1_acc']*100:.1f}%")
cols[2].metric("Deployed Acc", f"{meta['mit2_acc']*100:.1f}%")

st.divider()

uploaded = st.file_uploader(
    "Upload a face image (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col_result:
        with st.spinner("Running TTA inference (×5 views)…"):
            pred_idx, avg_probs = predict(model, image, meta, device)

        age_label = meta["age_names"][pred_idx]
        confidence = float(avg_probs[pred_idx]) * 100

        st.markdown(
            f"""
            <div class="result-card">
                <div class="prediction-label">🎂 {age_label}</div>
                <div class="confidence-pct">Confidence: {confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Class Probabilities")
    sorted_idx = np.argsort(avg_probs)[::-1]
    for i in sorted_idx:
        label = meta["age_names"][i]
        prob = float(avg_probs[i])
        is_top = i == pred_idx
        col_l, col_b = st.columns([1, 3])
        with col_l:
            if is_top:
                st.markdown(f"**{label}** ✅")
            else:
                st.markdown(label)
        with col_b:
            st.progress(prob, text=f"{prob*100:.1f}%")

    # Strategy info
    st.markdown(
        f"""
        <div class="info-box">
            <strong>Mitigation strategy:</strong><br>
            {meta['strategy']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Supported demographic groups (race):**")
    race_html = " ".join(
        f'<span class="pill">{r}</span>' for r in meta["race_names"]
    )
    st.markdown(race_html, unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="info-box">
        Upload any face photo above — the model will estimate the age group
        using 5-view Test-Time Augmentation for more robust predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "FairVisionCNN_v3 · Trained on FairFace 0.25 · "
    f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'} · "
    "9 age classes · 7 race groups · 2 gender groups"
)
