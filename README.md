 Crop Soil Prediction App

This project is a simple machine learning application that predicts the best crop to plant based on soil and environmental data. It uses a trained PyTorch model and a Streamlit web app for an interactive interface.

## Features

- Predicts suitable crops based on soil parameters
- Interactive UI built with Streamlit
- Uses a PyTorch model (`leaf_model.pth`) for predictions
- Easy to set up and run locally

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/monika-sundarapu/BOV
   cd CROP-SOIL-PREDECTION
   ```

2. Create and activate a virtual environment:

   - On Windows:
     ```
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install required packages:

   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app with:

```
streamlit run app1.py
```

Then open `http://localhost:8501` in your web browser to use the app.

## Project Structure

```
app1.py               # Streamlit web app
data_core.csv         # Dataset CSV file
leaf_model.pth        # Trained PyTorch model (not tracked in repo)
requirements.txt      # Python dependencies
venv/                 # Virtual environment folder (gitignored)
README.md             # Project documentation
.gitignore            # Files/folders ignored by git
```

## Notes

- The file `leaf_model.pth` (PyTorch model) is large and excluded from Git. You can add it using Git LFS or share separately if needed.
- Always keep your `venv/` folder in `.gitignore` to avoid committing the virtual environment.
- Feel free to customize and extend the project!

## Author

 
GitHub: [Monika sundarapu](https://github.com/monika-sundarapu/BOV)

If you want, I can help you create the `requirements.txt` file or assist with deployment! 
