# app.py
import streamlit as st

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Sugarcane Leaf Disease Analyzer")

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset # Not strictly needed for inference app, but transforms are
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt # For Grad-CAM overlay, not for st.pyplot
import cv2 # For Grad-CAM overlay

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import google.generativeai as genai
import time

# --- Configuration from your script (adapted for app) ---
# These must match the training configuration for the saved model
RANDOM_STATE = 42
IMG_SIZE = 256
CROP_SIZE = 224
MODEL_PATH = 'H:\\Sugarcane_Research\\Models\\Saved_models\\best_shufflenet_sugarcane.pth'
BASE_DATA_DIR = 'H:\\Sugarcane_Research\\Datasets\\'


# Predefined Best Hyperparameters (CRUCIAL for model loading)
BEST_HYPERPARAMS = {
    "dropout_rate1": 0.4312228339203463,
    "dropout_rate2": 0.21379023776628686,
    "freeze_ratio": 0.32297743904623005,
}

# Set random seeds
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Definition (Copied and adapted from your script) ---
class RegularizedShuffleNet(nn.Module):
    def __init__(self, num_classes, dropout_rate1=0.5, dropout_rate2=0.5, freeze_ratio=0.5):
        super().__init__()
        self.base_model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        all_params = list(self.base_model.parameters())
        learnable_params_indices = [i for i, p in enumerate(all_params) if p.requires_grad]
        num_learnable_params_to_freeze = int(len(learnable_params_indices) * freeze_ratio)
        if num_learnable_params_to_freeze > 0:
            for i in range(num_learnable_params_to_freeze):
                param_idx_to_freeze = learnable_params_indices[i]
                all_params[param_idx_to_freeze].requires_grad = False
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate1),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout_rate2),
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        return self.base_model(x)

# --- Helper Functions ---

@st.cache_data
def get_class_names_and_mapping():
    # This function might call st.error/st.stop, ensure it's called AFTER set_page_config
    if not os.path.exists(BASE_DATA_DIR):
        # If this runs at import time before set_page_config, it could be an issue.
        # However, direct calls to st.error/stop are the main concern for set_page_config.
        # The decorators themselves are fine before set_page_config.
        # The actual invocation of the function matters.
        st.error(f"Dataset directory not found: {BASE_DATA_DIR}. Please check the path.")
        st.stop() # This is a Streamlit command
    try:
        class_names = sorted([d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d))])
        if not class_names:
            st.error(f"No class subdirectories found in {BASE_DATA_DIR}.")
            st.stop() # This is a Streamlit command
        num_classes = len(class_names)
        class_to_int = {name: i for i, name in enumerate(class_names)}
        int_to_class = {i: name for i, name in enumerate(class_names)}
        return class_names, num_classes, class_to_int, int_to_class
    except Exception as e:
        st.error(f"Error reading class names from {BASE_DATA_DIR}: {e}")
        st.stop() # This is a Streamlit command

# Call this *after* st.set_page_config has definitely run.
# So, defining it here is fine, but its first execution should be within the main app flow.
CLASS_NAMES, NUM_CLASSES, CLASS_TO_INT, INT_TO_CLASS = get_class_names_and_mapping()


@st.cache_resource
def load_sugarcane_model():
    # This function also might call st.error/st.success etc.
    model = RegularizedShuffleNet(
        num_classes=NUM_CLASSES,
        dropout_rate1=BEST_HYPERPARAMS["dropout_rate1"],
        dropout_rate2=BEST_HYPERPARAMS["dropout_rate2"],
        freeze_ratio=BEST_HYPERPARAMS["freeze_ratio"]
    ).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        # st.success(f"Model '{MODEL_PATH}' loaded successfully on {device}.") # Moved to main app logic
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}. Ensure it's in the correct directory.")
        st.stop()
    except RuntimeError as e:
        st.error(f"Error loading model weights: {e}. "
                 "This often means the model architecture in app.py "
                 "doesn't match the one used to save '{MODEL_PATH}'. "
                 "Verify NUM_CLASSES and BEST_HYPERPARAMS.")
        st.info("Trying to load with weights_only=False (less secure)...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
            # st.success(f"Model '{MODEL_PATH}' loaded successfully with weights_only=False.") # Moved
        except Exception as e2:
            st.error(f"Still failed to load with weights_only=False: {e2}")
            st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        st.stop()
    model.eval()
    return model

# Call this *after* st.set_page_config
MODEL = load_sugarcane_model()
# Display model load success message here, now that set_page_config has run


# Define transformations
inference_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_pil):
    return inference_transforms(image_pil).unsqueeze(0).to(device)

def predict(image_tensor):
    with torch.no_grad():
        outputs = MODEL(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top5_prob, top5_indices = torch.topk(probabilities, 5, dim=1)
    top5_prob = top5_prob.squeeze().cpu().numpy()
    top5_indices = top5_indices.squeeze().cpu().numpy()
    predictions = []
    for i in range(len(top5_indices)):
        class_idx = top5_indices[i]
        class_name = INT_TO_CLASS.get(class_idx, "Unknown")
        confidence = top5_prob[i]
        predictions.append({"class_name": class_name, "confidence": float(confidence)})
    main_prediction_idx = top5_indices[0]
    main_prediction_name = INT_TO_CLASS.get(main_prediction_idx, "Unknown")
    main_confidence = top5_prob[0]
    return main_prediction_name, float(main_confidence), predictions, main_prediction_idx


def generate_grad_cam_image(model, input_tensor, original_pil_image, target_category_idx=None):
    target_layers = [model.base_model.conv5[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    if target_category_idx is not None:
        targets = [ClassifierOutputTarget(target_category_idx)]
    else:
        targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    img_for_overlay_pil = original_pil_image.resize((CROP_SIZE, CROP_SIZE))
    img_for_overlay_np = np.array(img_for_overlay_pil) / 255.0
    visualization = show_cam_on_image(img_for_overlay_np, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)


# --- Gemini API Integration ---
# This block also uses st.secrets and st.warning, ensure it's fine after set_page_config
GEMINI_AVAILABLE = False
GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Streamlit command
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    GEMINI_AVAILABLE = True
except (KeyError, AttributeError, NameError): # NameError if st.secrets doesn't exist
    # st.warning("Gemini API key not found in secrets.toml. Recommendation mode will be limited.") # Defer this warning
    pass # Will show warning later in the UI if needed
except Exception as e:
    # st.warning(f"Could not initialize Gemini: {e}. Recommendation mode will be limited.") # Defer this warning
    pass


def get_gemini_recommendations(disease_name):
    if not GEMINI_AVAILABLE or not gemini_model: # Add check for gemini_model
        return "Gemini AI is not available or not initialized. Recommendations cannot be generated."

    prompt = f"""
    The sugarcane leaf disease identified is: {disease_name}.
    In simple, easy-to-understand language suitable for a farmer, provide the following information:

    1.  **Findings (Why this disease might have occurred):** Briefly explain common causes or conditions that lead to {disease_name} in sugarcane.
    2.  **Preventive Actions (What to do now):** Suggest immediate, practical steps to manage the current situation and prevent further spread.
    3.  **Future Steps (Long-term prevention):** Recommend long-term strategies to avoid {disease_name} in future crops.

    Keep each section concise (2-4 bullet points or short paragraphs).
    Avoid overly technical jargon.
    """
    try:
        start_time = time.time()
        response = gemini_model.generate_content(prompt)
        end_time = time.time()
        st.sidebar.info(f"Gemini query took {end_time - start_time:.2f}s") # Streamlit command
        return response.text
    except Exception as e:
        st.error(f"Error getting recommendations from Gemini: {e}") # Streamlit command
        return "Could not retrieve recommendations at this time."

# --- Streamlit App UI ---
# st.set_page_config(layout="wide", page_title="Sugarcane Leaf Disease Analyzer") # MOVED TO TOP

st.title("🌿 Sugarcane Leaf Disease Analyzer")

# Show Gemini status warning here if needed, after title
if not GEMINI_AVAILABLE:
    st.warning("Gemini API key not found or Gemini failed to initialize. Recommendation mode will be limited.")


# Sidebar for mode selection and image upload
st.sidebar.header("Controls")
app_mode = st.sidebar.radio(
    "Choose Mode",
    ("Reference Model Analysis", "Disease Diagnosis & Recommendations")
)

uploaded_file = st.sidebar.file_uploader("Upload a Sugarcane Leaf Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original_image_pil = Image.open(uploaded_file).convert('RGB')
    
    # --- Analyze Image ---
    with st.spinner("Analyzing image..."):
        image_tensor = preprocess_image(original_image_pil)
        main_pred_name, main_confidence, top_preds, main_pred_idx = predict(image_tensor)

    # --- Display Images Side by Side ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        # Display original image, using use_container_width
        st.image(original_image_pil, caption="Original Uploaded Image", use_container_width=True)

    with col2:
        if app_mode == "Reference Model Analysis": # Only show Grad-CAM in this mode
            st.subheader("Grad-CAM Visualization")
            with st.spinner("Generating Grad-CAM..."):
                try:
                    grad_cam_img = generate_grad_cam_image(MODEL, image_tensor, original_image_pil, target_category_idx=main_pred_idx)
                    st.image(grad_cam_img, caption=f"Grad-CAM for {main_pred_name}", use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate Grad-CAM: {e}")
                    st.info("This can happen if the target layer is not found or if there are issues with the Grad-CAM library interaction.")
        else: # For "Disease Diagnosis & Recommendations" mode, col2 can be empty or show other info
            st.empty() # Or you could put a placeholder/title here if desired

    # --- Display Predictions and Recommendations Below Images ---
    st.markdown("---") # Horizontal line separator

    st.subheader("Analysis Results")
    st.success(f"**Main Prediction:** {main_pred_name} ({main_confidence*100:.2f}%)")

    if app_mode == "Reference Model Analysis":
        st.subheader("Top 5 Predictions")
        for pred in top_preds:
            st.text(f"{pred['class_name']}:")
            # Using st.progress for the visual bar
            progress_value = int(pred['confidence'] * 100)
            st.progress(progress_value, text=f"{pred['confidence']*100:.2f}%")


    elif app_mode == "Disease Diagnosis & Recommendations":
        st.subheader("Disease Information & Recommendations")
        if GEMINI_AVAILABLE:
            with st.spinner(f"Fetching recommendations for {main_pred_name} from Gemini AI..."):
                recommendations = get_gemini_recommendations(main_pred_name)
            st.markdown(recommendations)
        else:
            st.markdown(f"**Basic Info for {main_pred_name}:**\n"
                        f"- **Findings:** Consult agricultural extension for specific causes.\n"
                        f"- **Preventive Actions:** Isolate affected plants if possible. Improve air circulation.\n"
                        f"- **Future Steps:** Consider resistant varieties. Practice crop rotation.")
else:
    st.info("Awaiting image upload... Please upload an image using the sidebar.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** ShuffleNetV2 (Customized)")
st.sidebar.markdown(f"**Classes:** {NUM_CLASSES} ({', '.join(CLASS_NAMES[:3])}...)")
st.sidebar.markdown("---")
st.sidebar.markdown("")