import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ========== Page Setup ==========
st.set_page_config(page_title="Soil & Leaf Analyzer", layout="centered")
st.title("🌾 Smart Crop & Disease Advisor")
st.markdown("**Step 1: Predict Ideal Soil Conditions**  ➡️  **Step 2: Detect Leaf Disease**")

# ========== Load Leaf Disease Model ==========
@st.cache_resource
def load_leaf_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("leaf_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

leaf_model = load_leaf_model()
leaf_classes = ['healthy', 'powdery', 'rust']
leaf_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

def predict_leaf(image):
    image_tensor = leaf_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = leaf_model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
    return leaf_classes[pred.item()], confidence.item(), probs[0]

# ========== Load Crop Requirement Model ==========
@st.cache_data
def load_crop_data():
    df = pd.read_csv("data_core.csv")
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    le_fert = LabelEncoder()
    df["Crop Type"] = le_crop.fit_transform(df["Crop Type"])
    df["Soil Type"] = le_soil.fit_transform(df["Soil Type"])
    df["Fertilizer Name"] = le_fert.fit_transform(df["Fertilizer Name"])
    X = df[["Crop Type"]]
    y = df.drop(columns=["Crop Type"])
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le_crop, le_soil, le_fert

crop_model, le_crop, le_soil, le_fert = load_crop_data()

# ========== Step 1: Crop Requirements ==========
st.header("🟤 Step 1: Crop Requirement Predictor")
crop_name = st.selectbox("Select a Crop:", le_crop.classes_)

if st.button("Predict Ideal Soil Conditions"):
    try:
        encoded = le_crop.transform([crop_name])
        prediction = crop_model.predict([[encoded[0]]])[0]

        result = {
            "🌡️ Temperature (°C)": round(prediction[0], 2),
            "💧 Humidity (%)": round(prediction[1], 2),
            "🪴 Moisture (%)": round(prediction[2], 2),
            "🧪 Soil Type": le_soil.inverse_transform([int(round(prediction[3]))])[0],
            "🔋 Nitrogen (N)": int(round(prediction[4])),
            "🧬 Phosphorus (P)": int(round(prediction[5])),
            "⚗️ Potassium (K)": int(round(prediction[6])),
            "🧫 Recommended Fertilizer": le_fert.inverse_transform([int(round(prediction[7]))])[0],
        }

        st.success(f"Recommended Growing Conditions for **{crop_name}**")
        for key, val in result.items():
            st.write(f"{key}: **{val}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ========== Step 2: Leaf Disease Detection ==========
st.header("🌿 Step 2: Leaf Disease Detection")
st.markdown("Upload a leaf image or use the webcam to detect if it's **healthy**, **powdery**, or **rust** infected.")

mode = st.radio("Input Method:", ["📁 Upload Image", "📷 Use Webcam"])
image = None

if mode == "📁 Upload Image":
    file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file).convert("RGB")

elif mode == "📷 Use Webcam":
    cam_image = st.camera_input("Capture Leaf Photo")
    if cam_image:
        image = Image.open(cam_image).convert("RGB")

if image:
    st.image(image, caption="Leaf Image", use_column_width=True)
    label, conf, probs = predict_leaf(image)
    st.success(f"Prediction: **{label}** ({conf*100:.2f}% confidence)")

    st.subheader("📊 Class Probabilities")
    for i, p in enumerate(probs):
        st.write(f"- {leaf_classes[i]}: {p.item()*100:.2f}%")
