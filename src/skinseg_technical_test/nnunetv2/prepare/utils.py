import os
import SimpleITK as sitk
import numpy as np
from skinseg_technical_test.nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
from PIL import Image


def load_mask_as_array(mask_path,channel_to_labels= {0: 1, 1: 2, 2: 3},type='img'):
    """
    Loads an RGB PNG mask image and converts it to a numpy array.
    
    Parameters:
    - mask_path: Path to the RGB PNG mask file (str or Path object)
    
    Returns:
    - numpy array with shape (height, width, 3) for RGB, or (height, width) for grayscale
    """
    mask_path = Path(mask_path)
    
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Load the image
    img = Image.open(mask_path)
    
    # Convert to numpy array
    mask_array = np.array(img)
    if type == 'img':
        return mask_array

    # convert rgb [n,m,3] to [n,m] with values 0,1,2,3
    # Create an empty array for the label mask
    label_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)
        
    # Map RGB values to labels
    for channel, label in channel_to_labels.items():
        label_mask[mask_array[:, :, channel] > 0] = label
        
    return label_mask



def crop_to_common_shape(image_array, mask_array):
    """Crop image and mask to the smallest shared shape."""
    if image_array is None or mask_array is None:
        return image_array, mask_array

    image_shape = np.asarray(image_array).shape
    mask_shape = np.asarray(mask_array).shape

    if len(image_shape) < 2 or len(mask_shape) < 2:
        return image_array, mask_array

    h = min(image_shape[0], mask_shape[0])
    w = min(image_shape[1], mask_shape[1])

    image_array = image_array[:h, :w]
    mask_array = mask_array[:h, :w]

    return image_array, mask_array


def im_to_niftii(input_path, dest_name, resolution=None, type='img'):
    """
    Converts an image to NIfTI format and saves it to the specified destination.

    Parameters:
    - input: The input image (SimpleITK Image or NumPy array).
    - dest_name: The destination file name for the NIfTI image.
    - resolution: Optional. The desired resolution for the output image.
    - spacing: Optional. The desired spacing for the output image.
    """
    if isinstance(input_path, (str, Path)):
        img_array = load_mask_as_array(input_path, type=type)
    else:
        img_array = np.asarray(input_path)
    # Convert NumPy array to SimpleITK Image
    input_image = sitk.GetImageFromArray(img_array)
    if resolution is not None:
        input_image.SetSpacing(resolution)
    
    # Save the image in NIfTI format
    sitk.WriteImage(input_image, dest_name)

def make_out_dirs(dataset_id: int, task_name: str):
    """
    Creates the necessary output directories for the dataset.

    Parameters:
    - dataset_id: The ID of the dataset.
    - task_name: The name of the task.
    """

    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"
    out_dir = nnUNet_raw / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    imagesTr_dir = out_dir / "imagesTr"
    imagesTr_dir.mkdir(exist_ok=True)
    
    imagesTs_dir = out_dir / "imagesTs"
    imagesTs_dir.mkdir(exist_ok=True)
    
    labelsTr_dir = out_dir / "labelsTr"
    labelsTr_dir.mkdir(exist_ok=True)

    labelsTs_dir = out_dir / "labelsTs"
    labelsTs_dir.mkdir(exist_ok=True)
    
    preprocessed_dir = nnUNet_preprocessed / dataset_name
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, imagesTr_dir, labelsTr_dir, imagesTs_dir, labelsTs_dir, preprocessed_dir

def get_train_test_split(df: pd.DataFrame, test_size: float = 0.25, n_folds: int = 3, random_state: int = 3103):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - df_path: The path to the CSV file containing the dataset.
    - test_size: The proportion of the dataset to include in the test split.
    - n_folds: The number of folds for cross-validation.
    - random_state: Controls the shuffling applied to the data before applying the split.
    Returns:
    - Tuple containing the training and testing sets.
    """
    # Load dataframe
    patient_metadata = get_patient_metadata(df)
    # patient metadata
    patient_labelled = patient_metadata[patient_metadata["label_all"]]
    # Stratified split by region_canonical
    train_df_labelled , test_df = train_test_split(
        patient_labelled, 
        test_size=test_size, 
        random_state=random_state,
        stratify=patient_labelled['region_canonical']
    )
    
    # Ensure every category is in test_df
    missing_categories = set(patient_labelled['region_canonical'].unique()) - set(test_df['region_canonical'].unique())
    
    if missing_categories:
        print(f"Warning: Missing categories in test_df: {missing_categories}")
        print("Adding one sample per missing category...")
        
        for category in missing_categories:
            # Find a sample from the missing category in train_df
            category_samples = train_df_labelled[train_df_labelled['region_canonical'] == category]
            if len(category_samples) > 0:
                # Move one sample from train to test
                sample_idx = category_samples.index[0]
                test_df = pd.concat([test_df, train_df_labelled.loc[[sample_idx]]])
                train_df_labelled = train_df_labelled.drop(sample_idx)
    test_ids = test_df['patient_id'].tolist()
    print(f"Test set categories: {sorted(test_df['region_canonical'].unique())}")
    print(f"All categories represented: {len(test_df['region_canonical'].unique()) == len(patient_labelled['region_canonical'].unique())}")
    
    # Generate n-fold cross-validation splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_val_folds = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(
        train_df_labelled, 
        train_df_labelled['region_canonical']
    )):
        fold_dict = {
            'train': train_df_labelled.iloc[train_indices]['patient_id'].tolist() +  patient_metadata[patient_metadata["label_all"]==0]['patient_id'].tolist(),
            'val': train_df_labelled.iloc[val_indices]['patient_id'].tolist()
        }
        train_val_folds.append(fold_dict)
        print(f"Fold {fold_idx + 1}: {len(fold_dict['train'])} train, {len(fold_dict['val'])} val")
    return train_val_folds, test_ids, patient_metadata

def get_patient_metadata(df: pd.DataFrame, masks_path: str | None = None):
    """
    Extracts metadata for each patient from the DataFrame.

    Parameters:
    - df: The DataFrame containing the dataset.

    Returns:
    - A DataFrame containing metadata for each patient, region, and canonical region.
    """
    
    if masks_path is None:
        data_path = Path(
            os.environ.get(
                "SKINSEG_DATA_PATH",
                Path(__file__).resolve().parents[4] / "data",
            )
        )
        masks_path = data_path / "masks"

    # Replace NaN values in region column with 'Unknown'
    df = df.copy()
    df['region'] = df['region'].fillna('Unknown')
    patient_ids = df['patient'].unique()
    patient_df = []
    
    for patient_id in patient_ids:
        df_patient = df[df['patient'] == patient_id]
        regions = df_patient['region'].unique()
        for i, region in enumerate(regions):
            print(f"Processing patient {patient_id}, region {region} ({i+1}/{len(regions)})")
            df_region = df_patient[df_patient['region'] == region]
            # Extract metadata for the patient and region
            num_images = len(df_region)
            # Map to canonical region
            canonical_region = map_region_to_canonical(region)
            # get labels
            img_names = df_region['img_name'].values
            label_surface, label_jde, label_corneous = 0 , 0 , 0
            for img_name in img_names: 
                mask = Path(masks_path) / img_name
                # Load the image
                mask_array = load_mask_as_array(mask, type='img')
                label_surface = label_surface + 1 * (np.sum(mask_array[:, :, 0] > 0) > 0)
                label_jde = label_jde + 1 * (np.sum(mask_array[:, :, 1] ) > 0 )
                label_corneous = label_corneous + 1 * (np.sum(mask_array[:, :, 2]) > 0 )
            label_surface_tol = (label_surface / len(img_names)) > 0.5
            label_jde_tol = (label_jde / len(img_names)) > 0.5
            label_corneous_tol = (label_corneous / len(img_names)) > 0.5
            # Append metadata to the list
            patient_dict = {
                'patient_id': patient_id + "_" + str(i),
                'patient': patient_id,
                'region': region,
                'region_canonical': canonical_region,
                'num_images': num_images,
                'label_surface': label_surface_tol,
                'label_jde': label_jde_tol,
                'label_corneous': label_corneous_tol,
                'label_all': label_surface_tol and label_jde_tol and label_corneous_tol
            }
            print(patient_dict)
            patient_df.append(patient_dict)
    patient_metadata = pd.DataFrame(patient_df)
    return patient_metadata


def map_region_to_canonical(region: str) -> str:
    """
    Maps detailed anatomical region names to canonical body regions.
    
    Parameters:
    - region: The detailed region name (e.g., 'Left Anterior Scalp')
    
    Returns:
    - Canonical region name: 'Head/Face', 'Upper Limb', 'Lower Limb', 'Torso', or 'Unknown'
    """
    # Handle NaN/None/empty
    if pd.isna(region) or region is None or str(region).strip() == '':
        return 'Unknown'
    
    region_lower = str(region).lower().strip()
    
    # Head/Face regions
    head_keywords = ['scalp', 'forehead', 'nose', 'cheek', 'eyebrow', 'ear', 'temporal', 
                     'lip', 'chin', 'mandibular', 'preauricular', 'ala nasi']
    if any(keyword in region_lower for keyword in head_keywords):
        return 'Head/Face'
    
    # Upper limb regions
    upper_limb_keywords = ['arm', 'forearm', 'hand']
    if any(keyword in region_lower for keyword in upper_limb_keywords):
        return 'Upper Limb'
    
    # Lower limb regions
    lower_limb_keywords = ['leg', 'foot', 'ankle']
    if any(keyword in region_lower for keyword in lower_limb_keywords):
        return 'Lower Limb'
    
    # Torso regions
    torso_keywords = ['chest', 'abdomen', 'back', 'paraspinal', 'scapular', 'lumbosacral']
    if any(keyword in region_lower for keyword in torso_keywords):
        # merge small torso category into Head/Face to avoid rare-class split issues
        return 'Head/Face'
    
    return 'Unknown'