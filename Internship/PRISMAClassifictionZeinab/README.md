# PRISMA Hyperspectral Classification - Streamlit GUI

A comprehensive web-based interface for hyperspectral image classification using PRISMA satellite data and Vision Transformers (ViT).

## 🚀 Features

### 🛰️ **Data Preprocessing**
- **HDF5 to TIFF Conversion**: Convert PRISMA HDF5 files to GeoTIFF format
- **Shapefile Processing**: Convert training labels from shapefiles to raster format
- **Band Selection**: Customizable VNIR and SWIR band removal
- **Data Validation**: Automatic path and file existence checks

### 🤖 **Model Training**
- **Vision Transformer (ViT)**: State-of-the-art transformer architecture
- **Cross-Attention Fusion (CAF)**: Advanced attention mechanisms
- **Hyperparameter Tuning**: Interactive sliders for all model parameters
- **Real-time Monitoring**: Live training progress and metrics

### 🧪 **Model Testing & Evaluation**
- **Performance Metrics**: Overall Accuracy, Average Accuracy, Kappa Coefficient
- **Confusion Matrix**: Visual classification performance analysis
- **Prediction Maps**: Generate and visualize classification results
- **Batch Processing**: Configurable testing parameters

### 📊 **Visualization & Analysis**
- **Interactive Plots**: Plotly-based interactive visualizations
- **Classification Maps**: High-resolution map visualization
- **🌍 Google Satellite View**: Overlay results on satellite imagery
- **Training History**: Learning curves and convergence analysis
- **Data Statistics**: Comprehensive band and pixel analysis

### ⚙️ **Configuration & Settings**
- **Path Management**: Centralized configuration management
- **Hardware Settings**: GPU selection and optimization
- **Model Architecture**: Customizable transformer parameters
- **Validation Tools**: Automatic path and dependency checking

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PRISMAClassifictionZeinab
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import streamlit, torch, rasterio; print('All dependencies installed successfully!')"
   ```

5. **Test imports (optional but recommended)**
   ```bash
   python test_imports.py
   ```

## 🚀 Quick Start

### 1. Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### 2. Navigate Through the Interface

#### 🏠 **Home Page**
- Overview of the system
- Project status indicators
- Quick start guide

#### 📊 **Data Preprocessing**
- Upload and convert PRISMA HDF5 files
- Process training data shapefiles
- Prepare pixel-level training data

#### 🤖 **Model Training**
- Configure model architecture
- Set training parameters
- Monitor training progress

#### 🧪 **Model Testing**
- Evaluate trained models
- Generate predictions
- Analyze performance metrics

#### 📈 **Results & Visualization**
- View classification maps
- **🌍 Google Satellite visualization with classification overlay**
- Analyze performance metrics
- Explore training history

#### ⚙️ **Settings**
- Configure paths and parameters
- Hardware optimization
- Model architecture settings

## 🌍 Google Satellite Visualization

### **New Feature: Interactive Satellite View**

The application now includes a powerful Google Satellite visualization feature that allows you to:

#### **Base Map Options**
- **Google Satellite**: High-resolution satellite imagery (recommended)
- **Google Terrain**: Terrain and street view
- **OpenStreetMap**: Open-source street map
- **ESRI World Imagery**: Professional satellite imagery

#### **Classification Overlay**
- Overlay your classification results on satellite imagery
- Adjustable transparency (0.0 to 1.0)
- Multiple color schemes (viridis, plasma, inferno, magma, tab20, Set3)
- Real-time layer toggling

#### **Training Data Visualization**
- Highlight training areas with red markers
- Show class information on hover
- Sample training points to avoid overcrowding

#### **Interactive Features**
- Zoom and pan navigation
- Layer controls for toggling overlays
- Fullscreen mode
- Measurement tools
- Mini-map for navigation context
- Legend for class interpretation

#### **Export Options**
- Save interactive maps as HTML files
- Download maps for offline viewing
- Share maps with colleagues

### **Usage Instructions**
1. Go to **Results & Visualization** → **🌍 Google Satellite View**
2. Select your preferred base map
3. Choose the classification map to overlay
4. Adjust transparency and color scheme
5. Click "Generate Interactive Map"
6. Use layer controls to toggle different overlays
7. Zoom and pan to explore specific areas

## 📁 Project Structure

```
PRISMAClassifictionZeinab/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── src/                     # Source code
│   ├── hsmodel/            # Model implementations
│   │   ├── demo.py         # Original training script
│   │   └── vit_pytorch.py  # Vision Transformer model
│   ├── preprocessing/       # Data preprocessing modules
│   │   ├── converth5totiff.py
│   │   ├── convertshapelabelstoraster.py
│   │   ├── split_pixel.py
│   │   └── color_map.py
│   └── utils/              # Utility modules
│       └── config.py       # Configuration management
├── outputs/                 # Generated outputs
├── saved_models/           # Trained models
└── logs/                   # Training logs
```

## 🔧 Configuration

### Path Configuration
The application uses a centralized configuration system in `src/utils/config.py`:

```python
class Config:
    def __init__(self):
        self.base_data_dir = "path/to/your/data"
        self.tifpath = "path/to/input.tif"
        self.train_mask_path = "path/to/train_mask.tif"
        self.test_mask_path = "path/to/test_mask.tif"
        self.save_model_path = "path/to/save/models"
        self.output_tif_path = "path/to/output.tif"
```

### Model Parameters
Key configurable parameters include:
- **Patch Size**: Size of image patches (1-11)
- **Band Patches**: Number of related bands (1-5)
- **Transformer Depth**: Number of layers (1-12)
- **Attention Heads**: Number of attention heads (1-16)
- **Learning Rate**: Training learning rate (1e-6 to 1e-2)
- **Batch Size**: Training batch size (8-256)

## 📊 Data Formats

### Input Data
- **PRISMA HDF5**: `.he5` files with VNIR and SWIR data
- **Training Labels**: Shapefiles (`.shp`) with class information
- **Reference Raster**: GeoTIFF for georeference

### Output Data
- **Classification Maps**: GeoTIFF format with class labels
- **Performance Metrics**: Accuracy, Kappa, confusion matrix
- **Training Logs**: Detailed training history and parameters

## 🎯 Usage Examples

### Example 1: Complete Workflow
1. **Preprocess Data**: Upload HDF5 file and convert to TIFF
2. **Prepare Labels**: Convert shapefile to raster masks
3. **Train Model**: Configure and train ViT model
4. **Evaluate Results**: Test model and analyze performance
5. **Visualize**: Explore classification maps and metrics

### Example 2: Model Fine-tuning
1. **Load Existing Model**: Use pre-trained model
2. **Adjust Parameters**: Modify architecture or training settings
3. **Continue Training**: Resume training with new parameters
4. **Compare Results**: Analyze performance improvements

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're in the correct directory
cd PRISMAClassifictionZeinab

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### GPU Issues
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Set GPU device
export CUDA_VISIBLE_DEVICES=0
```

#### File Path Issues
- Verify all paths in `config.py` are correct
- Ensure directories exist and are writable
- Check file permissions

### Performance Optimization
- Use GPU acceleration for training
- Adjust batch size based on available memory
- Monitor system resources during training
- Use appropriate patch sizes for your data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PRISMA satellite mission for hyperspectral data
- Vision Transformer architecture research
- Streamlit community for the web framework
- Open-source geospatial libraries

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation

---

**Happy Classifying! 🛰️🤖📊** 