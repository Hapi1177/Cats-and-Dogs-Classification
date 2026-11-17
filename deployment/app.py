from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")

# Path relative to this file, so the app works from any working directory
MODEL_PATH = Path(__file__).parent / "resnet18_best.pt"


@st.cache_resource
def load_model():
    # Model was saved on GPU -> map to CPU for web hosting
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    class_to_idx = ckpt["class_to_idx"]                 # {'Cat': 0, 'Dog': 1}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    h, w = ckpt.get("image_size", (224, 224))
    tfm = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tfm, idx_to_class


model, tfm, idx_to_class = load_model()

st.title("🐾 Cat vs Dog Classifier")
st.write("Upload a photo and the ResNet18 model will predict whether it's a cat or a dog.")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width="stretch")

    with torch.no_grad():
        probs = torch.softmax(model(tfm(img).unsqueeze(0)), dim=1)[0]

    top = int(probs.argmax())
    st.success(f"Prediction: **{idx_to_class[top]}** ({probs[top]:.1%} confidence)")

    chart = pd.DataFrame({"Probability": probs.numpy()},
                         index=[idx_to_class[i] for i in range(len(probs))])
    st.bar_chart(chart)
