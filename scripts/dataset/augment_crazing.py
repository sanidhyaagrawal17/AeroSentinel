import os
import cv2
import albumentations as A
import shutil
from pathlib import Path

def augment_crazing(src_dir, dest_dir):
    # This script assumes dest_dir ALREADY contains a copy of the full dataset
    # It will find crazing images, augment them, and save them back into dest_dir
    src_images_dir = os.path.join(src_dir, 'images')
    src_labels_dir = os.path.join(src_dir, 'labels')
    
    dest_images_dir = os.path.join(dest_dir, 'images')
    dest_labels_dir = os.path.join(dest_dir, 'labels')
    
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)
    
    # Define heavy augmentations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.8),
        A.GaussNoise(p=0.5),
        A.Affine(rotate=(-15, 15), translate_percent=(0.1, 0.1), scale=(0.9, 1.1), p=0.8)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    all_images = [f for f in os.listdir(src_images_dir) if f.endswith(('.jpg', '.png'))]
    augmented_count = 0
    
    for img_file in all_images:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label_path = os.path.join(src_labels_dir, label_file)
        
        if not os.path.exists(src_label_path):
            continue
            
        # Check if it has crazing
        bboxes = []
        class_labels = []
        has_crazing = False
        
        with open(src_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id == 0:
                        has_crazing = True
                    # Albumentations expects [x_center, y_center, width, height]
                    bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    class_labels.append(cls_id)
                    
        if has_crazing:
            # Load image
            img_path = os.path.join(src_images_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate 2 augmented versions per crazing image
            for i in range(2):
                try:
                    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_img = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']
                    
                    if len(aug_bboxes) == 0:
                        continue # Augmentation might have cropped out the bbox
                        
                    # Save new image and label
                    new_img_name = f"aug_{i}_" + img_file
                    new_label_name = f"aug_{i}_" + label_file
                    
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(dest_images_dir, new_img_name), aug_img_bgr)
                    
                    with open(os.path.join(dest_labels_dir, new_label_name), 'w') as f:
                        for bbox, cls_id in zip(aug_bboxes, aug_labels):
                            f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                            
                    augmented_count += 1
                except Exception as e:
                    print(f"Error augmenting {img_file}: {e}")
                    
    print(f"Generated {augmented_count} new augmented crazing images in {dest_dir}!")

if __name__ == "__main__":
    print("Augmenting Crazing dataset...")
    # Assume data/crazing_augmented already has train/ and valid/ copied to it
    # And we want to read from train/ and valid/ and save augmented images into data/crazing_augmented/
    augment_crazing("train", "data/crazing_augmented/train")
    augment_crazing("valid", "data/crazing_augmented/valid")
