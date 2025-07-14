import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("leaf_model.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ['healthy', 'powdery', 'rust']

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict_image_with_confidence(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_pct = confidence.item() * 100

    # Show
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class} ({confidence_pct:.2f}%)")
    plt.axis("off")
    plt.show()

    print(f"\n✅ Prediction: {predicted_class}")
    print("📊 Class Probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  - {class_names[i]}: {prob.item()*100:.2f}%")

# Example usage
predict_image_with_confidence("capture.jpg")
