import os

class Config:
    def __init__(self):
        # === Base data directory ===
        self.base_data_dir = r"Internship"
        self.base_data_outputs = os.path.join(self.base_data_dir, 'outputs')

        # === Original HDF5 scene information ===
        self.scene_id = "PRS_L2D_STD_20200725083506_20200725083510_0001.he5"
        self.scene_path = os.path.join(self.base_data_dir, self.scene_id)
        self.h5_path = self.scene_id
        self.h5_vnir_key = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"
        self.h5_swir_key = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"
        self.h5_lat_key = "HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude"
        self.h5_lon_key = "HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude"
        
        # === Band removal settings ===
        self.vnir_bands_to_remove = [1, 2, 3, 4, 5]
        self.swir_bands_to_remove = list(range(3, 6)) + list(range(40, 57)) + list(range(86, 112)) + list(range(152, 172))
        
        # === Input raster files (processed data) ===
        self.tifpath = os.path.join(self.base_data_dir, 'outputs', 'vnir_swir_stackedd.tif')
        self.train_mask_path = os.path.join(self.base_data_dir, 'outputs', 'train_mask.tif')
        self.test_mask_path = os.path.join(self.base_data_dir, 'outputs', 'test_mask.tif')
        self.label_tif_path = os.path.join(self.base_data_dir, 'outputs', 'colored_class_labels.tif')
        self.shapefile_path = os.path.join(self.base_data_dir, 'Lithium paper', 'Lithium_Train.shp')
        
        # === Output directories ===
        self.save_model_path = os.path.join(self.base_data_dir, 'saved_models')
        self.final_log_path = os.path.join(self.base_data_dir, 'logs')
        self.output_tif_path = os.path.join(self.base_data_dir, 'outputs', 'final_prediction_map.tif')
        
        # === Output files ===
        self.matrix_output_path = 'matrix.mat'
        self.new_final_class_labels = os.path.join(self.base_data_dir, 'outputs', 'new__newfinal_class_labels.tif')
        self.colored_output_tif = os.path.join(self.base_data_dir, 'outputs', 'colored_class_labels.tif')
        
        # === Class field settings ===
        self.class_field = 'Classcode'  # Adjust this to match your shapefile's class field name
        self.class_names = ['Background','AFG','BGr','GMGr','GTG','Mgb','MGr','MPGr','Oph','PGn','WD']

        # === Create necessary directories if not exist ===
        self._create_required_dirs()

    def _create_required_dirs(self):
        # Collect all directories from the defined paths
        dirs = {
            os.path.dirname(self.tifpath),
            os.path.dirname(self.train_mask_path),
            os.path.dirname(self.test_mask_path),
            os.path.dirname(self.label_tif_path),
            os.path.dirname(self.shapefile_path),
            os.path.dirname(self.save_model_path),
            os.path.dirname(self.final_log_path),
            os.path.dirname(self.output_tif_path),
            os.path.dirname(self.new_final_class_labels),
            os.path.dirname(self.colored_output_tif),
        }

        # Create each directory
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

config = Config()
