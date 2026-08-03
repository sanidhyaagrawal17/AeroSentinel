import os
import glob
import shutil

def main():
    splits = ['train', 'valid', 'test']
    base_in = 'data/processed'
    base_out = 'data/crazing_only'

    total_extracted = 0

    for split in splits:
        in_labels_dir = os.path.join(base_in, split, 'labels')
        in_images_dir = os.path.join(base_in, split, 'images')
        
        out_labels_dir = os.path.join(base_out, split, 'labels')
        out_images_dir = os.path.join(base_out, split, 'images')
        
        os.makedirs(out_labels_dir, exist_ok=True)
        os.makedirs(out_images_dir, exist_ok=True)
        
        labels = glob.glob(os.path.join(in_labels_dir, '*.txt'))
        
        for lbl_path in labels:
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            # Keep only class 0 (crazing)
            crazing_lines = [line for line in lines if line.startswith('0 ')]
            
            if len(crazing_lines) > 0:
                # Write new label file
                filename = os.path.basename(lbl_path)
                out_lbl_path = os.path.join(out_labels_dir, filename)
                with open(out_lbl_path, 'w') as f:
                    f.writelines(crazing_lines)
                
                # Copy image
                # Need to find the image file. Usually it's .jpg
                img_name = filename.replace('.txt', '.jpg')
                in_img_path = os.path.join(in_images_dir, img_name)
                out_img_path = os.path.join(out_images_dir, img_name)
                
                if os.path.exists(in_img_path):
                    shutil.copy(in_img_path, out_img_path)
                    total_extracted += 1
                else:
                    print(f"Warning: Image {img_name} not found!")

    print(f"\nExtraction Complete: {total_extracted} pure Crazing images extracted to {base_out}!")

    # Write the YAML for this new dataset
    yaml_content = f"""path: ../data/crazing_only
train: train/images
val: valid/images
test: test/images

names:
  0: crazing
"""
    with open('split_crazing_only.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Created 'split_crazing_only.yaml'")

if __name__ == "__main__":
    main()
