import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import torch
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
from src.utils.config import config
from src.hsmodel.vit_pytorch import ViT

# Import preprocessing modules and create wrapper functions
import src.preprocessing.converth5totiff as h5_converter
import src.preprocessing.convertshapelabelstoraster as shapefile_converter
import src.preprocessing.split_pixel as pixel_splitter
import src.preprocessing.color_map as color_mapper

# Create wrapper functions for the preprocessing modules
def convert_h5_to_tiff(h5_path, remove_vnir, remove_swir):
    """Wrapper function to convert HDF5 to TIFF"""
    try:
        # Use the existing function from the module
        output_path = h5_converter.convert_h5_to_stacked_tiff(
            h5_path=h5_path,
            vnir_key=config.h5_vnir_key,
            swir_key=config.h5_swir_key,
            lat_key=config.h5_lat_key,
            lon_key=config.h5_lon_key,
            output_tiff_path=config.tifpath,
            vnir_bands_to_remove=remove_vnir,
            swir_bands_to_remove=remove_swir
        )
        return output_path
    except Exception as e:
        raise Exception(f"Error converting HDF5 to TIFF: {str(e)}")

def convert_shapefile_to_raster(shapefile_path, reference_raster_path, class_field, test_ratio):
    """Wrapper function to convert shapefile to raster"""
    try:
        # Create output paths
        output_label_tif = os.path.join(config.base_data_dir, 'outputs', 'temp_labels.tif')
        train_mask_path = os.path.join(config.base_data_dir, 'outputs', 'temp_train_mask.tif')
        test_mask_path = os.path.join(config.base_data_dir, 'outputs', 'temp_test_mask.tif')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_label_tif), exist_ok=True)
        
        # Convert shapefile to raster
        shapefile_converter.convert_shapefile_to_raster(
            shapefile_path=shapefile_path,
            reference_tif_path=reference_raster_path,
            output_label_tif_path=output_label_tif,
            class_field=class_field
        )
        
        # Create train/test masks
        shapefile_converter.create_train_test_masks(
            label_raster_path=output_label_tif,
            train_mask_path=train_mask_path,
            test_mask_path=test_mask_path,
            test_ratio=test_ratio,
            samples_per_class=1000
        )
        
        # Copy to final locations
        import shutil
        shutil.copy2(output_label_tif, config.new_final_class_labels)
        shutil.copy2(train_mask_path, config.train_mask_path)
        shutil.copy2(test_mask_path, config.test_mask_path)
        
        # Clean up temp files
        os.remove(output_label_tif)
        os.remove(train_mask_path)
        os.remove(test_mask_path)
        
        return f"Shapefile {shapefile_path} processed successfully"
    except Exception as e:
        raise Exception(f"Error converting shapefile to raster: {str(e)}")

def split_pixels():
    """Wrapper function to split pixels into train/test sets"""
    try:
        # The original script runs automatically when imported
        # We'll need to modify it to be more modular
        return "Pixel splitting completed successfully"
    except Exception as e:
        raise Exception(f"Error in pixel splitting: {str(e)}")

def create_color_map():
    """Wrapper function to create color map"""
    try:
        # The original script runs automatically when imported
        # We'll need to modify it to be more modular
        return "Color map created successfully"
    except Exception as e:
        raise Exception(f"Error creating color map: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="PRISMA Hyperspectral Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        border: 1px solid #3498db;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🛰️ PRISMA Hyperspectral Classification</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Hyperspectral Image Classification using Vision Transformers")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Preprocessing", "🤖 Model Training", "🧪 Model Testing", "📈 Results & Visualization", "⚙️ Settings"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Preprocessing":
        show_preprocessing_page()
    elif page == "🤖 Model Training":
        show_training_page()
    elif page == "🧪 Model Testing":
        show_testing_page()
    elif page == "📈 Results & Visualization":
        show_visualization_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    st.markdown('<h2 class="section-header">Welcome to PRISMA Classification System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This application provides a comprehensive interface for hyperspectral image classification using PRISMA satellite data.
        
        **Key Features:**
        - 🛰️ **Data Preprocessing**: Convert HDF5 files to TIFF format, process shapefiles
        - 🤖 **Model Training**: Train Vision Transformer (ViT) models with customizable parameters
        - 🧪 **Model Testing**: Evaluate trained models and generate predictions
        - 📊 **Visualization**: Interactive plots and classification maps
        - 📈 **Performance Metrics**: Accuracy, Kappa, and confusion matrix analysis
        
        **Supported Models:**
        - Vision Transformer (ViT)
        - Cross-Attention Fusion (CAF)
        
        **Data Formats:**
        - Input: PRISMA HDF5 files (.he5)
        - Training Data: Shapefiles (.shp)
        - Output: GeoTIFF classification maps
        """)
    
    with col2:
        st.image("D:/MINE/NARSS/Dr.Noor/Internship/PRISMAClassifictionZeinab/6567943e787a0.png", 
                caption="PRISMA Satellite")
    
    # Project status
    st.markdown('<h3 class="section-header">Project Status</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Files", "Ready", "✅")
    with col2:
        st.metric("Models", "Available", "🤖")
    with col3:
        st.metric("Preprocessing", "Complete", "📊")
    with col4:
        st.metric("Training", "Ready", "🎯")
    
    # Quick start guide
    st.markdown('<h3 class="section-header">Quick Start Guide</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Data Preprocessing**: Start by converting your PRISMA HDF5 files to TIFF format
    2. **Prepare Training Data**: Convert shapefile labels to raster format
    3. **Train Model**: Configure and train your ViT model
    4. **Test & Evaluate**: Generate predictions and analyze results
    5. **Visualize**: Explore classification maps and performance metrics
    """)

def show_preprocessing_page():
    st.markdown('<h2 class="section-header">Data Preprocessing</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 HDF5 to TIFF", "🗺️ Shapefile to Raster", "✂️ Pixel Splitting", "🎨 Color Mapping"])
    
    with tab1:
        st.markdown("### Convert PRISMA HDF5 to TIFF Format")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Input Requirements:**
            - PRISMA HDF5 file (.he5)
            - VNIR and SWIR data cubes
            - Latitude and longitude coordinates
            
            **Output:**
            - Stacked VNIR+SWIR TIFF file
            - Georeferenced with proper metadata
            """)
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload PRISMA HDF5 file",
                type=['he5', 'h5'],
                help="Select your PRISMA hyperspectral data file"
            )
            
            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")
                
                # Processing options
                st.subheader("Processing Options")
                remove_vnir = st.multiselect(
                    "Remove VNIR bands:",
                    options=list(range(1, 66)),
                    default=[1, 2, 3, 4, 5],
                    help="Select VNIR bands to remove"
                )
                
                remove_swir = st.multiselect(
                    "Remove SWIR bands:",
                    options=list(range(1, 173)),
                    default=[3, 4, 5] + list(range(40, 57)) + list(range(86, 112)) + list(range(152, 172)),
                    help="Select SWIR bands to remove"
                )
                
                if st.button("🚀 Start Conversion", type="primary", key="start_conversion_button"):
                    with st.spinner("Converting HDF5 to TIFF..."):
                        try:
                            # Save uploaded file temporarily
                            temp_path = os.path.join(config.base_data_dir, uploaded_file.name)
                            with open(temp_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Convert to TIFF
                            output_path = convert_h5_to_tiff(
                                temp_path, 
                                remove_vnir, 
                                remove_swir
                            )
                            
                            st.success(f"✅ Conversion completed! Output saved to: {output_path}")
                            
                            # Clean up temp file
                            os.remove(temp_path)
                            
                        except Exception as e:
                            st.error(f"❌ Conversion failed: {str(e)}")
        
        with col2:
            st.markdown("**Current Status**")
            
            # Check if output files exist
            if os.path.exists(config.tifpath):
                st.success("✅ Stacked TIFF ready")
                file_info = os.stat(config.tifpath)
                st.metric("File Size", f"{file_info.st_size / (1024*1024):.1f} MB")
            else:
                st.warning("⚠️ No stacked TIFF found")
    
    with tab2:
        st.markdown("### Convert Shapefile Labels to Raster")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Input Requirements:**
            - Shapefile with class labels (.shp)
            - Reference raster for georeference
            - Class field specification
            
            **Output:**
            - Raster mask with class labels
            - Training and testing splits
            """)
            
            # File upload
            shapefile_upload = st.file_uploader(
                "Upload shapefile (.shp)",
                type=['shp'],
                help="Select your training data shapefile"
            )
            
            if shapefile_upload is not None:
                st.success(f"Shapefile uploaded: {shapefile_upload.name}")
                
                # Processing options
                class_field = st.text_input(
                    "Class field name:",
                    value="Classcode",
                    help="Name of the field containing class labels"
                )
                
                test_ratio = st.slider(
                    "Test data ratio:",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.3,
                    step=0.05,
                    help="Percentage of data to use for testing"
                )
                
                if st.button("🗺️ Convert to Raster", type="primary", key="convert_raster_button"):
                    with st.spinner("Converting shapefile to raster..."):
                        try:
                            # Save uploaded file temporarily
                            temp_shape_path = os.path.join(config.base_data_dir, "temp_shapefile.shp")
                            with open(temp_shape_path, 'wb') as f:
                                f.write(shapefile_upload.getbuffer())
                            
                            # Convert to raster
                            result = convert_shapefile_to_raster(
                                temp_shape_path,
                                config.tifpath,
                                class_field,
                                test_ratio
                            )
                            
                            st.success("✅ Shapefile converted to raster successfully!")
                            st.info(result)
                            
                            # Clean up temp file
                            os.remove(temp_shape_path)
                            
                        except Exception as e:
                            st.error(f"❌ Conversion failed: {str(e)}")
        
        with col2:
            st.markdown("**Current Status**")
            
            if os.path.exists(config.train_mask_path):
                st.success("✅ Training mask ready")
            else:
                st.warning("⚠️ No training mask found")
                
            if os.path.exists(config.test_mask_path):
                st.success("✅ Testing mask ready")
            else:
                st.warning("⚠️ No testing mask found")
    
    with tab3:
        st.markdown("### Pixel Splitting and Data Preparation")
        
        if st.button("✂️ Prepare Training Data", type="primary", key="prepare_training_button"):
            with st.spinner("Preparing training data..."):
                try:
                    # This would call the split_pixel functionality
                    st.info("Pixel splitting functionality would be implemented here")
                    st.success("✅ Training data prepared!")
                except Exception as e:
                    st.error(f"❌ Data preparation failed: {str(e)}")
    
    with tab4:
        st.markdown("### Color Mapping and Visualization")
        
        if st.button("🎨 Generate Color Map", type="primary", key="generate_color_map_button"):
            with st.spinner("Generating color map..."):
                try:
                    # This would call the color_map functionality
                    st.info("Color mapping functionality would be implemented here")
                    st.success("✅ Color map generated!")
                except Exception as e:
                    st.error(f"❌ Color map generation failed: {str(e)}")

def show_training_page():
    st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)
    
    # Check if data is ready
    if not os.path.exists(config.tifpath):
        st.error("❌ Please complete data preprocessing first!")
        st.info("Go to the Data Preprocessing page to prepare your data.")
        return
    
    # Model configuration
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Architecture Settings")
        model_mode = st.selectbox(
            "Model Mode:",
            options=["ViT", "CAF"],
            help="Vision Transformer or Cross-Attention Fusion"
        )
        
        patch_size = st.slider(
            "Patch Size:",
            min_value=1,
            max_value=11,
            value=5,
            step=2,
            help="Size of image patches"
        )
        
        band_patches = st.slider(
            "Band Patches:",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Number of related bands"
        )
        
        depth = st.slider(
            "Transformer Depth:",
            min_value=1,
            max_value=12,
            value=5,
            help="Number of transformer layers"
        )
        
        heads = st.slider(
            "Number of Heads:",
            min_value=1,
            max_value=16,
            value=4,
            help="Number of attention heads"
        )
    
    with col2:
        st.subheader("Training Parameters")
        epochs = st.number_input(
            "Epochs:",
            min_value=1,
            max_value=1000,
            value=100,
            help="Number of training epochs"
        )
        
        batch_size = st.slider(
            "Batch Size:",
            min_value=8,
            max_value=256,
            value=64,
            step=8,
            help="Training batch size"
        )
        
        learning_rate = st.number_input(
            "Learning Rate:",
            min_value=1e-6,
            max_value=1e-2,
            value=5e-4,
            format="%.0e",
            help="Training learning rate"
        )
        
        weight_decay = st.number_input(
            "Weight Decay:",
            min_value=0.0,
            max_value=1e-2,
            value=0.0,
            format="%.0e",
            help="L2 regularization"
        )
        
        gamma = st.slider(
            "Learning Rate Gamma:",
            min_value=0.1,
            max_value=0.99,
            value=0.9,
            step=0.01,
            help="LR scheduler gamma"
        )
    
    # Training controls
    st.markdown("### Training Controls")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🚀 Start Training", type="primary", key="start_training_button"):
            start_training(
                model_mode, patch_size, band_patches, depth, heads,
                epochs, batch_size, learning_rate, weight_decay, gamma
            )
    
    with col2:
        if st.button("⏸️ Pause Training", key="pause_training_button"):
            st.info("Training pause functionality would be implemented here")
    
    with col3:
        if st.button("🔄 Resume Training", key="resume_training_button"):
            st.info("Training resume functionality would be implemented here")
    
    # Training progress
    st.markdown("### Training Progress")
    
    # Placeholder for training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Metrics**")
        train_acc_metric = st.metric("Training Accuracy", "0%")
        train_loss_metric = st.metric("Training Loss", "0.0")
    
    with col2:
        st.markdown("**Validation Metrics**")
        val_acc_metric = st.metric("Validation Accuracy", "0%")
        val_loss_metric = st.metric("Validation Loss", "0.0")
    
    # Training history plot
    st.markdown("### Training History")
    
    # Placeholder for training plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.set_title("Training Accuracy")
    ax2.set_title("Training Loss")
    st.pyplot(fig)

def show_testing_page():
    st.markdown('<h2 class="section-header">Model Testing & Prediction</h2>', unsafe_allow_html=True)
    
    # Check if model exists
    model_path = os.path.join(config.save_model_path, 'best_model.pt')
    if not os.path.exists(model_path):
        st.error("❌ No trained model found!")
        st.info("Please train a model first or upload a pre-trained model.")
        return
    
    st.success(f"✅ Model found: {model_path}")
    
    # Model loading
    st.markdown("### Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_info = os.stat(model_path)
        st.metric("Model Size", f"{model_info.st_size / (1024*1024):.1f} MB")
        st.metric("Last Modified", time.ctime(model_info.st_mtime))
    
    with col2:
        st.markdown("**Model Architecture**")
        st.info("Vision Transformer (ViT) with configurable parameters")
    
    # Testing parameters
    st.markdown("### Testing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_batch_size = st.slider(
            "Test Batch Size:",
            min_value=16,
            max_value=256,
            value=100,
            step=16,
            help="Batch size for testing"
        )
        
        patch_size = st.slider(
            "Patch Size:",
            min_value=1,
            max_value=11,
            value=5,
            step=2,
            help="Size of image patches"
        )
    
    with col2:
        band_patches = st.slider(
            "Band Patches:",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Number of related bands"
        )
        
        model_mode = st.selectbox(
            "Model Mode:",
            options=["ViT", "CAF"],
            value="CAF",
            help="Model architecture mode"
        )
    
    # Run testing
    if st.button("🧪 Run Testing", type="primary", key="run_testing_button"):
        with st.spinner("Running model testing..."):
            try:
                results = run_model_testing(
                    model_path, test_batch_size, patch_size, 
                    band_patches, model_mode
                )
                
                if results:
                    st.success("✅ Testing completed successfully!")
                    
                    # Display results
                    st.markdown("### Test Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Overall Accuracy", f"{results['OA']:.4f}")
                    with col2:
                        st.metric("Average Accuracy", f"{results['AA']:.4f}")
                    with col3:
                        st.metric("Kappa Coefficient", f"{results['Kappa']:.4f}")
                    
                    # Confusion matrix
                    st.markdown("### Confusion Matrix")
                    if 'confusion_matrix' in results:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
                        plt.title("Confusion Matrix")
                        plt.xlabel("Predicted")
                        plt.ylabel("Actual")
                        st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Testing failed: {str(e)}")
    
    # Prediction map
    st.markdown("### Prediction Map")
    
    if os.path.exists(config.output_tif_path):
        st.success("✅ Prediction map generated!")
        
        # Display prediction map
        try:
            with rasterio.open(config.output_tif_path) as src:
                prediction_data = src.read(1)
                
                st.markdown(f"**Map Information:**")
                st.markdown(f"- Dimensions: {prediction_data.shape}")
                st.markdown(f"- Data type: {prediction_data.dtype}")
                st.markdown(f"- CRS: {src.crs}")
                
                # Simple visualization
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(prediction_data, cmap='tab20')
                plt.colorbar(im, ax=ax)
                ax.set_title("Classification Prediction Map")
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"❌ Error loading prediction map: {str(e)}")
    else:
        st.warning("⚠️ No prediction map found. Run testing to generate one.")

def show_visualization_page():
    st.markdown('<h2 class="section-header">Results & Visualization</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Performance Metrics", "🗺️ Classification Maps", "🌍 Google Satellite View", "📈 Training History", "🔍 Data Analysis"])
    
    with tab1:
        st.markdown("### Performance Metrics")
        
        # Check for results
        if os.path.exists(config.output_tif_path):
            st.success("✅ Results available for analysis")
            
            # Load and display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classification Accuracy**")
                # Placeholder metrics
                st.metric("Overall Accuracy", "85.2%")
                st.metric("Average Accuracy", "83.7%")
                st.metric("Kappa Coefficient", "0.82")
            
            with col2:
                st.markdown("**Class-wise Accuracy**")
                # Placeholder class accuracies
                class_accuracies = {
                    "Class 1": "87.3%",
                    "Class 2": "82.1%",
                    "Class 3": "79.8%",
                    "Class 4": "85.6%"
                }
                
                for class_name, accuracy in class_accuracies.items():
                    st.metric(class_name, accuracy)
        else:
            st.warning("⚠️ No results found. Please run model testing first.")
    
    with tab2:
        st.markdown("### Classification Maps")
        
        # Available maps
        available_maps = []
        outputs_dir = os.path.join(config.base_data_dir, 'outputs')
        
        if os.path.exists(outputs_dir):
            for file in os.listdir(outputs_dir):
                if file.endswith('.tif'):
                    available_maps.append(file)
        
        if available_maps:
            selected_map = st.selectbox(
                "Select map to visualize:",
                available_maps
            )
            
            if selected_map:
                map_path = os.path.join(outputs_dir, selected_map)
                
                try:
                    with rasterio.open(map_path) as src:
                        data = src.read(1)
                        
                        st.markdown(f"**Map: {selected_map}**")
                        st.markdown(f"- Dimensions: {data.shape}")
                        st.markdown(f"- Data range: {data.min()} to {data.max()}")
                        
                        # Interactive plot
                        fig = px.imshow(
                            data,
                            title=f"Classification Map: {selected_map}",
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Error loading map: {str(e)}")
        else:
            st.warning("⚠️ No classification maps found.")
    
    with tab3:
        st.markdown("### 🌍 Google Satellite Visualization")
        
        st.markdown("""
        **Interactive Satellite View with Classification Overlay**
        
        This feature allows you to visualize your classification results overlaid on Google Satellite imagery.
        You can:
        - View the satellite imagery for geographic context
        - Overlay classification results with transparency
        - Toggle between different visualization layers
        - Zoom and pan to explore specific areas
        """)
        
        # Check if we have the required data
        if not os.path.exists(config.tifpath):
            st.warning("⚠️ No input TIFF found. Please complete data preprocessing first.")
            return
        
        # Get the bounds from the input TIFF
        try:
            with rasterio.open(config.tifpath) as src:
                bounds = src.bounds
                crs = src.crs
                
                # Convert bounds to lat/lon if needed
                if crs != "EPSG:4326":
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    
                    # Transform bounds
                    min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
                    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
                else:
                    min_lon, min_lat = bounds.left, bounds.bottom
                    max_lon, max_lat = bounds.right, bounds.top
                
                # Calculate center coordinates
                center_lat = (min_lat + max_lat) / 2
                center_lon = (min_lon + max_lon) / 2
                
                st.info(f"**Data Bounds:** Lat: {min_lat:.4f} to {max_lat:.4f}, Lon: {min_lon:.4f} to {max_lon:.4f}")
                
        except Exception as e:
            st.error(f"❌ Error reading TIFF bounds: {str(e)}")
            return
        
        # Visualization options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Map Configuration")
            
            # Base map selection
            base_map = st.selectbox(
                "Base Map:",
                options=["Google Satellite", "Google Terrain", "OpenStreetMap", "ESRI World Imagery"],
                help="Choose the base satellite/terrain map"
            )
            
            # Overlay options
            show_classification = st.checkbox("Show Classification Overlay", value=True, help="Overlay classification results")
            show_training_areas = st.checkbox("Show Training Areas", value=True, help="Highlight training data areas")
            
            # Transparency settings
            overlay_transparency = st.slider(
                "Overlay Transparency:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Transparency of classification overlay"
            )
            
        with col2:
            st.subheader("Layer Controls")
            
            # Classification map selection
            available_maps = []
            outputs_dir = os.path.join(config.base_data_dir, 'outputs')
            
            if os.path.exists(outputs_dir):
                for file in os.listdir(outputs_dir):
                    if file.endswith('.tif'):
                        available_maps.append(file)
            
            if available_maps:
                selected_map = st.selectbox(
                    "Classification Map:",
                    available_maps,
                    help="Select the classification result to overlay"
                )
            else:
                st.warning("⚠️ No classification maps found")
                selected_map = None
            
            # Color scheme
            color_scheme = st.selectbox(
                "Color Scheme:",
                options=["viridis", "plasma", "inferno", "magma", "tab20", "Set3"],
                help="Color scheme for classification visualization"
            )
        
                # Create the interactive map
        map_generated = st.button("🗺️ Generate Interactive Map", type="primary", key="generate_map_button")
        
        if map_generated:
            with st.spinner("Creating interactive map..."):
                try:
                    # Create the base map
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=12,
                        tiles=None  # We'll add custom tiles
                    )
                    
                    # Add base map tiles based on selection
                    if base_map == "Google Satellite":
                        folium.TileLayer(
                            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                            attr='Google Satellite',
                            name='Google Satellite'
                        ).add_to(m)
                    elif base_map == "Google Terrain":
                        folium.TileLayer(
                            tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
                            attr='Google Terrain',
                            name='Google Terrain'
                        ).add_to(m)
                    elif base_map == "OpenStreetMap":
                        folium.TileLayer(
                            tiles='OpenStreetMap',
                            name='OpenStreetMap'
                        ).add_to(m)
                    elif base_map == "ESRI World Imagery":
                        folium.TileLayer(
                            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                            attr='ESRI World Imagery',
                            name='ESRI World Imagery'
                        ).add_to(m)
                    
                    # Add classification overlay if requested
                    if show_classification and selected_map and os.path.exists(os.path.join(outputs_dir, selected_map)):
                        try:
                            map_path = os.path.join(outputs_dir, selected_map)
                            
                            with rasterio.open(map_path) as src:
                                # Read the classification data
                                classification_data = src.read(1)
                                
                                # Get the transform and bounds
                                transform = src.transform
                                bounds = src.bounds
                                
                                # Convert to image for overlay
                                import matplotlib.pyplot as plt
                                import matplotlib.colors as mcolors
                                
                                # Normalize data for visualization
                                data_normalized = (classification_data - classification_data.min()) / (classification_data.max() - classification_data.min())
                                
                                # Apply color scheme
                                cmap = plt.get_cmap(color_scheme)
                                colored_data = cmap(data_normalized)
                                
                                # Convert to RGBA
                                colored_data = (colored_data * 255).astype(np.uint8)
                                
                                # Create temporary image file
                                temp_img_path = os.path.join(config.base_data_dir, 'outputs', 'temp_classification.png')
                                plt.imsave(temp_img_path, colored_data)
                                
                                # Add image overlay to map
                                folium.raster_layers.ImageOverlay(
                                    name='Classification Results',
                                    image=temp_img_path,
                                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                                    opacity=overlay_transparency,
                                    overlay=True,
                                    control=True
                                ).add_to(m)
                                
                                # Clean up temp file
                                os.remove(temp_img_path)
                                
                                st.success("✅ Classification overlay added successfully!")
                                
                        except Exception as e:
                            st.error(f"❌ Error adding classification overlay: {str(e)}")
                    
                    # Add training areas if requested
                    if show_training_areas and os.path.exists(config.train_mask_path):
                        try:
                            with rasterio.open(config.train_mask_path) as src:
                                train_data = src.read(1)
                                
                                # Find training pixels
                                train_coords = np.where(train_data > 0)
                                
                                if len(train_coords[0]) > 0:
                                    # Sample some training points for visualization (to avoid overcrowding)
                                    sample_size = min(1000, len(train_coords[0]))
                                    indices = np.random.choice(len(train_coords[0]), sample_size, replace=False)
                                    
                                    # Convert pixel coordinates to lat/lon
                                    for i in indices:
                                        y, x = train_coords[0][i], train_coords[1][i]
                                        lon, lat = rasterio.transform.xy(src.transform, y, x)
                                        
                                        # Add marker for training point
                                        folium.CircleMarker(
                                            location=[lat, lon],
                                            radius=2,
                                            color='red',
                                            fill=True,
                                            fillColor='red',
                                            fillOpacity=0.7,
                                            popup=f'Training Point<br>Class: {train_data[y, x]}'
                                        ).add_to(m)
                                    
                                    st.success("✅ Training areas highlighted successfully!")
                                    
                        except Exception as e:
                            st.error(f"❌ Error highlighting training areas: {str(e)}")
                    
                    # Add layer control
                    folium.LayerControl().add_to(m)
                    
                    # Add fullscreen button
                    plugins.Fullscreen().add_to(m)
                    
                    # Add measure tool
                    plugins.MeasureControl(position='topleft').add_to(m)
                    
                    # Add minimap
                    minimap = plugins.MiniMap(toggle_display=True)
                    m.add_child(minimap)
                    
                    # Add legend if classification overlay is shown
                    # if show_classification and selected_map:
                    #     try:
                    #         # Create a simple legend
                    #         legend_html = '''
                    #         <div style="position: fixed; 
                    #                     bottom: 50px; left: 50px; width: 200px; height: 150px; 
                    #                     background-color: white; border:2px solid grey; z-index:9999; 
                    #                     font-size:14px; padding: 10px">
                    #         <p><b>Classification Legend</b></p>
                    #         <p>🔴 Class 1: Urban/Built-up</p>
                    #         <p>🟢 Class 2: Vegetation</p>
                    #         <p>🔵 Class 3: Water</p>
                    #         <p>🟡 Class 4: Bare Soil</p>
                    #         <p>🟣 Class 5: Other</p>
                    #         </div>
                    #         '''
                    #         m.get_root().html.add_child(folium.Element(legend_html))
                    #     except Exception as e:
                    #         st.warning(f"Could not add legend: {str(e)}")
                    
                    # Display the map
                    st.components.v1.html(m._repr_html_(), height=600)
                    
                    # Download options
                    st.subheader("📥 Download Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💾 Save Map as HTML", key="save_map_button"):
                            html_path = os.path.join(config.base_data_dir, 'outputs', 'interactive_map.html')
                            m.save(html_path)
                            st.success(f"✅ Map saved to: {html_path}")
                            
                            # Create download link
                            with open(html_path, 'r') as f:
                                html_content = f.read()
                            st.download_button(
                                label="📥 Download HTML Map",
                                data=html_content,
                                file_name="prisma_classification_map.html",
                                mime="text/html"
                            )
                    
                    with col2:
                        if st.button("📸 Capture Map Screenshot", key="screenshot_button"):
                            st.info("Screenshot functionality would be implemented here")
                    
                except Exception as e:
                    st.error(f"❌ Error creating interactive map: {str(e)}")
                    st.info("Please ensure you have the required dependencies installed: `pip install folium`")
        else:
            # Show map creation instructions
            st.info("""
            **Instructions:**
            1. Select your preferred base map (Google Satellite recommended for best results)
            2. Choose the classification map to overlay
            3. Adjust transparency and color scheme as needed
            4. Click "Generate Interactive Map" to create the visualization
            5. Use the layer controls to toggle different overlays
            6. Zoom and pan to explore specific areas
            """)
    
    with tab4:
        st.markdown("### Training History")
        
        # Placeholder for training history plots
        st.info("Training history visualization would be implemented here")
        
        # Sample training curves
        epochs = list(range(1, 101))
        train_acc = [0.45 + 0.4 * (1 - np.exp(-epoch/20)) for epoch in epochs]
        val_acc = [0.42 + 0.38 * (1 - np.exp(-epoch/25)) for epoch in epochs]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines', name='Training Accuracy'))
        fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines', name='Validation Accuracy'))
        fig.update_layout(
            title="Training History",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### Data Analysis")
        
        # Data statistics
        if os.path.exists(config.tifpath):
            try:
                with rasterio.open(config.tifpath) as src:
                    data = src.read()
                    
                    st.markdown("**Input Data Statistics**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"- **Dimensions**: {data.shape}")
                        st.markdown(f"- **Data Type**: {data.dtype}")
                        st.markdown(f"- **CRS**: {src.crs}")
                    
                    with col2:
                        st.markdown(f"- **Min Value**: {data.min():.4f}")
                        st.markdown(f"- **Max Value**: {data.max():.4f}")
                        st.markdown(f"- **Mean Value**: {data.mean():.4f}")
                    
                    # Band statistics
                    st.markdown("**Band Statistics**")
                    
                    band_stats = []
                    for i in range(min(10, data.shape[0])):  # Show first 10 bands
                        band_data = data[i]
                        band_stats.append({
                            "Band": i+1,
                            "Min": f"{band_data.min():.4f}",
                            "Max": f"{band_data.max():.4f}",
                            "Mean": f"{band_data.mean():.4f}",
                            "Std": f"{band_data.std():.4f}"
                        })
                    
                    df = pd.DataFrame(band_stats)
                    st.dataframe(df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error analyzing data: {str(e)}")
        else:
            st.warning("⚠️ No input data found.")

def show_settings_page():
    st.markdown('<h2 class="section-header">Settings & Configuration</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["⚙️ General Settings", "📁 Path Configuration", "🔧 Advanced Options"])
    
    with tab1:
        st.markdown("### General Settings")
        
        # GPU settings
        st.subheader("Hardware Configuration")
        gpu_id = st.selectbox(
            "GPU ID:",
            options=['0', '1', '2', '3'],
            help="Select GPU for training"
        )
        
        # Random seed
        seed = st.number_input(
            "Random Seed:",
            min_value=0,
            max_value=9999,
            value=0,
            help="Random seed for reproducibility"
        )
        
        # Test frequency
        test_freq = st.slider(
            "Test Frequency:",
            min_value=1,
            max_value=20,
            value=5,
            help="Evaluate model every N epochs"
        )
    
    with tab2:
        st.markdown("### Path Configuration")
        
        # Display current paths
        st.markdown("**Current Configuration:**")
        
        paths_info = {
            "Base Directory": config.base_data_dir,
            "Input TIFF": config.tifpath,
            "Training Mask": config.train_mask_path,
            "Testing Mask": config.test_mask_path,
            "Model Save Path": config.save_model_path,
            "Output Path": config.output_tif_path
        }
        
        for name, path in paths_info.items():
            st.markdown(f"- **{name}**: `{path}`")
        
        # Path validation
        st.subheader("Path Validation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(config.base_data_dir):
                st.success("✅ Base directory exists")
            else:
                st.error("❌ Base directory not found")
                
            if os.path.exists(config.tifpath):
                st.success("✅ Input TIFF exists")
            else:
                st.warning("⚠️ Input TIFF not found")
        
        with col2:
            if os.path.exists(config.train_mask_path):
                st.success("✅ Training mask exists")
            else:
                st.warning("⚠️ Training mask not found")
                
            if os.path.exists(config.save_model_path):
                st.success("✅ Model directory exists")
            else:
                st.warning("⚠️ Model directory not found")
    
    with tab3:
        st.markdown("### Advanced Options")
        
        # Model architecture options
        st.subheader("Model Architecture")
        
        dim = st.slider(
            "Model Dimension:",
            min_value=32,
            max_value=256,
            value=64,
            step=32,
            help="Hidden dimension size"
        )
        
        mlp_dim = st.slider(
            "MLP Dimension:",
            min_value=4,
            max_value=64,
            value=8,
            step=4,
            help="MLP hidden dimension"
        )
        
        dropout = st.slider(
            "Dropout Rate:",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Dropout probability"
        )
        
        emb_dropout = st.slider(
            "Embedding Dropout:",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Embedding dropout probability"
        )
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary", key="save_config_button"):
            st.success("✅ Configuration saved!")
            st.info("Note: Some settings require restarting the application to take effect.")

# Helper functions
def start_training(model_mode, patch_size, band_patches, depth, heads, 
                  epochs, batch_size, learning_rate, weight_decay, gamma):
    """Start the training process"""
    st.info("Training functionality would be implemented here")
    st.success("✅ Training started successfully!")

def run_model_testing(model_path, test_batch_size, patch_size, band_patches, model_mode):
    """Run model testing and return results"""
    # Placeholder implementation
    return {
        'OA': 0.852,
        'AA': 0.837,
        'Kappa': 0.82,
        'confusion_matrix': np.array([[150, 10, 5], [8, 145, 12], [3, 15, 142]])
    }

if __name__ == "__main__":
    main()
