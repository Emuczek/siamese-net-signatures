import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import glob
import shutil

DATA_DIR = 'data/signatures'
ORIGINAL_PATH = os.path.join(DATA_DIR, 'full_org')
FORGERY_PATH = os.path.join(DATA_DIR, 'full_forg')
PROCESSED_DATA_DIR = 'data/signatures_processsed'

IMG_WIDTH = 224
IMG_HEIGHT = 224


def preprocess_image(image_path):
    """Reading image, normalized and scaled to (WIDHT, HEIGHT) in grayscale"""
    
    try:
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None
        
        img_resized = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_normalized = img_resized.astype('float32') / 255.0

        return img_normalized
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def create_dataset_structure():
    """Creating folder structure for processed images."""

    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)

    os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'test'), exist_ok=True)

    print("Created processed data directory structure.")

def prepare_and_save_data():
    """Main loop for preprocessing and saving images"""

    create_dataset_structure()
    
    original_files = glob.glob(os.path.join(ORIGINAL_PATH, '*.png'))
    if not original_files:
        print(f"Error: No original signature files found in {ORIGINAL_PATH}. Make sure the path is correct.")
        return

    user_signatures = {}

    for file_path in original_files:
        filename = os.path.basename(file_path)
        user_id = int(filename.split('_')[1])
        
        if user_id not in user_signatures:
            user_signatures[user_id] = {'original': [], 'forged': []}
        
        user_signatures[user_id]['original'].append(file_path)

    forged_files = glob.glob(os.path.join(FORGERY_PATH, '*.png'))

    for file_path in forged_files:
        filename = os.path.basename(file_path)
        user_id = int(filename.split('_')[1])
        if user_id in user_signatures:
            user_signatures[user_id]['forged'].append(file_path)

    user_ids = list(user_signatures.keys())
    train_users, test_users = train_test_split(user_ids, test_size=0.2, random_state=42)

    print(f"Total users: {len(user_ids)}")
    print(f"Training users: {len(train_users)}")
    print(f"Testing users: {len(test_users)}")

    for user_id in user_ids:
        dataset_type = 'train' if user_id in train_users else 'test'

        for img_path in user_signatures[user_id]['original']:
            img = preprocess_image(img_path)
            if img is not None:
                save_dir = os.path.join(PROCESSED_DATA_DIR, dataset_type, str(user_id), 'original')
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.basename(img_path)
                save_path = os.path.join(save_dir, filename)
                cv.imwrite(save_path, (img * 255).astype(np.uint8))

        for img_path in user_signatures[user_id]['forged']:
            img = preprocess_image(img_path)
            if img is not None:
                save_dir = os.path.join(PROCESSED_DATA_DIR, dataset_type, str(user_id), 'forged')
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.basename(img_path)
                save_path = os.path.join(save_dir, filename)
                cv.imwrite(save_path, (img * 255).astype(np.uint8))
                
    print("Dataset preparation complete. Processed data saved to 'processed_data' directory.")

if __name__ == '__main__':
    prepare_and_save_data()