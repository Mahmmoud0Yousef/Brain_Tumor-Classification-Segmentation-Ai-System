# 🧠 Brain Tumor AI System

This system provides both **Classification** and **Segmentation** capabilities for brain tumor detection using deep learning in a **single application**.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Run Application


```bash
# Run the main application
streamlit run app.py
```

The application provides both Classification and Segmentation features in a single interface.

## 📁 Project Structure

```
Project Computer Vision/
├── app.py                          # All-in-One Streamlit app (Classification + Segmentation)
├── requirements.txt                # Python dependencies
├── Classification.ipynb            # Classification training
├── Segmentation.ipynb              # Segmentation training
├── model-brain-mri-ResNet50V2.h5  # Classification model
├── model-brain-mri-Unet.h5        # Segmentation model
├── test_images/                    # Classification test images
```

## 🔧 Features

### Single Application (`app.py`) - All-in-One Solution
- **Unified Interface**: Both Classification and Segmentation in one app
- **Binary Classification**: Tumor vs No Tumor detection
- **Confidence Scores**: Probability of tumor presence
- **Pixel-level Segmentation**: Precise tumor region detection
- **Multiple Outputs**: Binary mask, probability map, overlay
- **Statistics**: Tumor coverage percentage
- **Visual Results**: Color-coded tumor regions
- **Image Upload**: Support for JPG, PNG, BMP formats
- **Real-time Analysis**: Instant results
- **Mode Switching**: Toggle between Classification and Segmentation modes

## 📊 Model Performance

### Classification (ResNet50V2)
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~88%
- **Test Accuracy**: ~92%

### Segmentation (U-Net)
- **Architecture**: U-Net with skip connections
- **Input Size**: 256x256 pixels
- **Output**: Binary segmentation mask
- **Loss Function**: Binary Crossentropy + Dice Loss

## 🎯 Usage Guide

### Using the All-in-One Application:
1. Run `streamlit run app.py`
2. Upload a brain MRI image
3. Switch between Classification and Segmentation modes using the sidebar
4. View comprehensive results and analysis
5. Both models work with the same uploaded image

## ⚠️ Important Notes

1. **Medical Disclaimer**: These are AI screening tools and should not replace professional medical diagnosis
2. **Model Requirements**: Ensure model files are in the correct location
3. **Image Quality**: Use clear, high-quality MRI images for best results
4. **File Formats**: Support for JPG, PNG, BMP, TIFF formats

## 🐛 Troubleshooting

### Common Issues:

1. **Model Loading Error**:
   - Check if model files exist: `model-brain-mri-ResNet50V2.h5`, `model-brain-mri-Unet.h5`
   - Verify file permissions

2. **Segmentation Issues**:
   - Ensure images are in correct format
   - Check model compatibility
   - Try different image sizes

3. **Classification Issues**:
   - Adjust classification threshold
   - Check image preprocessing
   - Verify model training

## 📚 Dataset Sources

### Primary Datasets Used:
- **Classification Dataset**: [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Segmentation Dataset**: [LGG Brain MRI Segmentation (Kaggle)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

## 📈 Data Information

### Classification Data:
- **Total Images**: 253
- **Tumor Images (YES)**: 155
- **No Tumor Images (NO)**: 98

### Segmentation Data:
- **Source**: Kaggle Brain MRI Segmentation Dataset
- **Format**: TIFF files with corresponding masks
- **Patients**: Multiple patient directories
- **Images per Patient**: Variable (10-50 images)

## 🔍 Advanced Usage

### Custom Thresholds:
- Modify classification threshold in the app
- Adjust segmentation threshold in code
- Fine-tune for your specific use case

### Batch Processing:
- Modify the code for batch image processing
- Add support for multiple image uploads
- Implement automated reporting

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

