import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load CSV file
file_path = 'data_core.csv'  # Ensure this CSV is in the same folder
df = pd.read_csv(file_path)

# Label encoding
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_fert = LabelEncoder()

df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'])

# Features and labels
X = df[['Crop Type']]
y = df.drop(columns=['Crop Type'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel Trained Successfully!")
print(f"Mean Absolute Error: {mae:.2f}\n")

# Available crops
crop_options = list(le_crop.classes_)
print("Available Crops:")
for i, crop in enumerate(crop_options):
    print(f"{i + 1}. {crop}")

# User input
try:
    choice = int(input("\nEnter the number of your crop choice: "))
    crop_name = crop_options[choice - 1]
    crop_encoded = le_crop.transform([crop_name])

    # Predict
    prediction = model.predict([[crop_encoded[0]]])[0]

    # Decode and display result
    result = {
        "Temperature (°C)": round(prediction[0], 2),
        "Humidity (%)": round(prediction[1], 2),
        "Moisture (%)": round(prediction[2], 2),
        "Soil Type": le_soil.inverse_transform([int(round(prediction[3]))])[0],
        "Nitrogen (N)": int(round(prediction[4])),
        "Phosphorus (P)": int(round(prediction[5])),
        "Potassium (K)": int(round(prediction[6])),
        "Recommended Fertilizer": le_fert.inverse_transform([int(round(prediction[7]))])[0],
    }

    print(f"\nRecommended Conditions for '{crop_name}':")
    for key, value in result.items():
        print(f"{key}: {value}")

except Exception as e:
    print(f"\nError: {e}")
