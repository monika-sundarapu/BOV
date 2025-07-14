import streamlit as st

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="Leaf Disease Classifier", layout="centered")

import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import io

# ========== Load Model ==========
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("leaf_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
class_names = ['healthy', 'powdery', 'rust']

# ========== Image Transform ==========
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ========== Prediction Function ==========
def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, 1)
    return class_names[pred_class.item()], confidence.item(), probs[0]

# ========== Streamlit UI ==========
st.title("🌿 Leaf Disease Classifier")
st.markdown("Upload a leaf image or use your webcam to detect if it's healthy, rust-infected, or powdery.")

# Choose mode
mode = st.radio("Choose input method:", ["📁 Upload Image", "📷 Use Webcam"])

image = None

if mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif mode == "📷 Use Webcam":
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# Predict if image is loaded
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)
    st.markdown("### 🔍 Prediction:")
    label, conf, probs = predict(image)
    st.success(f"✅ Predicted: {label} ({conf*100:.2f}% confidence)")

    st.markdown("### 📊 Class Probabilities:")
    for i, prob in enumerate(probs):
        st.write(f"• {class_names[i]}: {prob.item()*100:.2f}%")
