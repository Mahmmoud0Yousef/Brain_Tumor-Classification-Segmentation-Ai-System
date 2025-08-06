import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import cv2
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Brain Tumor AI System",
    page_icon="🧠",
    layout="wide"
)

# --- Mode Selection ---
st.sidebar.title("🔬 Select Mode")
mode = st.sidebar.radio("Choose Task:", ["Classification", "Segmentation"], index=0)

# Title and description
st.title("🧠 Brain Tumor AI System")
st.markdown("""
<div style='color:#4F8BF9; font-size:20px;'>
    Upload a brain MRI image to detect a tumor (Classification) or segment tumor area (Segmentation).
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if mode == "Classification":
    # Load the model
    @st.cache_resource
    def load_model():
        """Load the trained model"""
        try:
            # Try to load the best model first
            model_path = "model_Check_Points/resnet50v2_best.h5"
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                st.success("✅ Model loaded successfully!")
                return model
            else:
                # Fallback to the main model
                model_path = "model-brain-mri-ResNet50V2.h5"
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path)
                    st.success("✅ Model loaded successfully!")
                    return model
                else:
                    st.error("❌ Model file not found!")
                    return None
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None

    # Preprocess image function
    def preprocess_image(image, target_size=(224, 224)):
        """Preprocess image for model prediction"""
        try:
            # Convert PIL image to numpy array
            if isinstance(image, Image.Image):
                img_array = img_to_array(image)
            else:
                img_array = image
                
            # Resize image
            img_resized = cv2.resize(img_array, target_size)
            
            # Convert to RGB if grayscale
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            elif img_resized.shape[2] == 1:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            
            # Normalize pixel values
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return None

    # Prediction function
    def predict_tumor(model, image):
        """Predict tumor presence in the image"""
        try:
            # Preprocess the image
            processed_img = preprocess_image(image)
            if processed_img is None:
                return None, None
            
            # Make prediction
            prediction = model.predict(processed_img, verbose=0)
            
            # Get probability and class
            probability = prediction[0][0]
            predicted_class = "Tumor Detected" if probability > 0.5 else "No Tumor"
            
            return predicted_class, probability
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None

    # Load model
    model = load_model()

    if model is not None:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
            help="Upload a brain MRI image in JPG, PNG, BMP, or TIF format"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded MRI Image", use_container_width=True)
            
            with col2:
                st.subheader("🔍 Analysis Results")
                
                # Make prediction
                predicted_class, probability = predict_tumor(model, image)
                
                if predicted_class is not None:
                    # Display results with better styling
                    if predicted_class == "Tumor Detected":
                        st.error(f"🚨 **{predicted_class}**")
                        st.markdown(f"**Confidence:** {probability:.2%}")
                        st.markdown("⚠️ **Please consult a medical professional immediately!**")
                    else:
                        st.success(f"✅ **{predicted_class}**")
                        st.markdown(f"**Confidence:** {probability:.2%}")
                        st.markdown("✅ **No tumor detected in this image**")
                    
                    # Progress bar for confidence
                    try:
                        st.progress(float(probability))
                    except Exception as e:
                        st.info(f"Confidence: {probability:.2%}")
                    
                    # Additional information
                    st.markdown("---")
                    st.markdown("**Model Information:**")
                    st.markdown(f"- **Model Type:** ResNet50V2")
                    st.markdown(f"- **Training Accuracy:** ~95%")
                    st.markdown(f"- **Validation Accuracy:** ~88%")
                    
                    # Disclaimer
                    st.warning("""
                    ⚠️ **Medical Disclaimer:**
                    This is an AI-powered screening tool and should not replace professional medical diagnosis. 
                    Always consult with qualified healthcare professionals for accurate diagnosis and treatment.
                    """)
                else:
                    st.error("❌ Failed to analyze the image. Please try again.")
        
        else:
            # Show sample images
            st.markdown("---")
            st.subheader("📋 Sample Images")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Tumor Present (YES)**")
                st.markdown("Expected: Tumor Detected")
            
            with col2:
                st.markdown("**No Tumor (NO)**")
                st.markdown("Expected: No Tumor")
            
            with col3:
                st.markdown("**Upload your image**")
                st.markdown("Get instant results")

    else:
        st.error("""
        ❌ **Model Loading Failed**
        
        Please ensure the model files are in the correct location:
        - `model_Check_Points/resnet50v2_best.h5` or
        - `model-brain-mri-ResNet50V2.h5`
        
        If the problem persists, please check the model training notebook.
        """)

else:
    st.subheader("🧠 Tumor Segmentation (Unet)")
    st.markdown("Upload a brain MRI image to segment the tumor area using Unet model.")
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image for segmentation...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        key="segmentation_uploader",
        help="Upload a brain MRI image in JPG, PNG, BMP, or TIF format"
    )
    @st.cache_resource
    def load_unet_model():
        try:
            model_path = "model-brain-mri-Unet.h5"
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path, compile=False)
                st.success("✅ Unet model loaded successfully!")
                return model
            else:
                st.error("❌ Unet model file not found!")
                return None
        except Exception as e:
            st.error(f"❌ Error loading Unet model: {str(e)}")
            return None
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file).convert("RGB")
            image = image.resize((256, 256))
            st.image(image, caption="Uploaded MRI Image (Resized)", use_container_width=True)
        with col2:
            st.subheader("🧩 Segmentation Mask")
            unet_model = load_unet_model()
            if unet_model is not None:
                # Preprocess image for Unet
                img_array = np.array(image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)  # (1, 256, 256, 3)
                try:
                    pred_mask = unet_model.predict(img_array)
                    # Post-process mask
                    mask = pred_mask[0]
                    if mask.shape[-1] == 1:
                        mask = mask[..., 0]
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    mask_img = Image.fromarray(mask).convert("L").resize((256, 256))
                    st.image(mask_img, caption="Predicted Mask", use_container_width=True, clamp=True)
                except Exception as e:
                    st.error(f"❌ Error during segmentation: {str(e)}")
            else:
                st.info("Segmentation will appear here after model integration.")
    else:
        st.info("Please upload an MRI image for segmentation.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🧠 Brain Tumor AI System | Classification & Segmentation</p>
</div>
""", unsafe_allow_html=True) 