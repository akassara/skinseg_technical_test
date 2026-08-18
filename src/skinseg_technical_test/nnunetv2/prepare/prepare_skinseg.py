import os
import logging
import numpy as np
from pathlib import Path
import pandas as pd

from batchgenerators.utilities.file_and_folder_operations import save_json
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from skinseg_technical_test.nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from skinseg_technical_test.nnunetv2.prepare.utils import (
    crop_to_common_shape,
    im_to_niftii,
    load_mask_as_array,
    make_out_dirs,
    get_train_test_split,
)


data_path = Path("/Users/amynkassara/Desktop/projects/skinseg_technical_test/data")
# ---------------------
# Logging setup
# ---------------------
def setup_logging(log_dir: Path):
    """
    Sets up logging to both console and a log file.

    Parameters:
    - log_dir: The directory where the log file will be saved.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "prepare_skinseg.log"
    
    # Root loggers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 

    # Avoid duplicate
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler) 
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

# ---------------------
# Main function
# ---------------------

def prepare_skinseg(dataset_id: int, task_name: str, test_size: float, n_folds: int, regenerate: bool = False):
    """
    Prepares the skin segmentation dataset for nnU-Net.

    Parameters:
    - dataset_id: The ID of the dataset.
    - task_name: The name of the task.
    """
    # Setup logging
    out_dir, imagesTr_dir, labelsTr_dir, imagesTs_dir, preprocessed_dir = make_out_dirs(dataset_id, task_name)
    setup_logging(preprocessed_dir / "logs")
    logging.info(f"Starting dataset preparation for task '{task_name}' with dataset ID {dataset_id}.")
    df = pd.read_csv(data_path / 'labels.csv')
    train_val, test, patient_metadata = get_train_test_split(df, test_size, n_folds)
    df = pd.merge(
    left=df, 
    right=patient_metadata,
    how='left',
    left_on=['patient', 'region'],
    right_on=['patient', 'region'],
)
    # Load your dataset (this is a placeholder; replace with actual loading logic)
    # For example, you might load images and labels from a CSV or a directory structure
    # Here we assume you have a DataFrame with columns 'image_path' and 'label_path'
    
    # Example DataFrame (replace with actual data loading)
    data_df = pd.DataFrame({
        'image_path': ['path/to/image1.nii', 'path/to/image2.nii'],
        'label_path': ['path/to/label1.nii', 'path/to/label2.nii']
    })

    # Process each image and label
    train_pids = 0
    test_pids = 0
    for idx, row in patient_metadata.iterrows():
        patient_id = row["patient_id"]
        patient = row["patient"]
        df_img = df[df['patient_id'] == patient_id]
        img_names = df_img['img_name'].values
        for img_name in img_names:
            img_path = data_path / 'imgs' / img_name
            label_path = data_path / 'masks' / f"{img_name.split('.png')[0]}.png"     
            if patient_id in test:
                img_dest = imagesTs_dir / f"{img_name.split('.png')[0]}_0000.nii.gz"
                if img_dest.exists() and not regenerate:
                    logging.info(f"Test image {img_dest} already exists. Skipping conversion.")
                    continue
                img_array = load_mask_as_array(img_path, type='img')
                img_array, _ = crop_to_common_shape(img_array, np.zeros_like(img_array))
                # Convert to NIfTI and save in imagesTs directory
                im_to_niftii(img_array, img_dest)
                test_pids += 1
                logging.info(f"Processed test image {img_name} for patient {patient_id}")   
            else:
                # Convert to NIfTI and save in imagesTr and labelsTr directories
                img_dest = imagesTr_dir / f"{img_name.split('.png')[0]}_0000.nii.gz"
                label_dest = labelsTr_dir / f"{img_name.split('.png')[0]}.nii.gz"
                if img_dest.exists() and label_dest.exists() and not regenerate:
                    logging.info(f"Train image {img_dest} and label {label_dest} already exist. Skipping conversion.")
                    continue
                img_array = load_mask_as_array(img_path, type='img')
                label_array = load_mask_as_array(label_path, type='mask')
                img_array, label_array = crop_to_common_shape(img_array, label_array)
                im_to_niftii(img_array, img_dest)
                im_to_niftii(label_array, label_dest, type='mask')
                train_pids += 1
                logging.info(f"Processed train image {img_name} for patient {patient_id}")    
    # dump json and csv
    save_json(train_val, preprocessed_dir / "splits_final.json")
    patient_metadata.to_csv(preprocessed_dir / "patient_metadata.csv", index=False)
    df[df['patient_id'].isin(test)].to_csv(preprocessed_dir / "test.csv", index=False)
    # Generate dataset JSON
    label_map = {'background': 0, 'surface': 1, 'jde': 2, 'corneous': 3, "ignore": 4}
    channel_names = {0: 'lc_oct'}
    print(train_pids)
    print(test_pids)
    generate_dataset_json(out_dir, channel_names=channel_names, labels=label_map,
                 num_training_cases=train_pids, file_ending=".nii.gz", 
                 dataset_name=f"Dataset{dataset_id:03d}_{task_name}")

    logging.info(f"Dataset preparation for task '{task_name}' completed.")


if __name__ == "__main__":
    # Example usage
    prepare_skinseg(dataset_id=1, task_name="SkinSegmentation", test_size= 0.25, n_folds = 3, regenerate = True)