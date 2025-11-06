import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["train", "test"], default="train", required=True, type=str)
    args = parser.parse_args()
    ################
    # Options
    ################
    # Input dir and output dir
    if (args.dataset == "train"):
        img_dir = 'data/train/left'
        output_dir = 'data/train/est_segmentation'
        sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']
    else:
        img_dir = 'data/test/left'
        output_dir = 'data/test/est_segmentation'
        sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################
    
    # Use the yolo11x-seg model for best accuracy, with 62.1M parameters
    model = YOLO("yolo/yolo11x-seg.pt")
    
    for sample_name in sample_list:
        img_path = os.path.join(img_dir, sample_name + ".png")
        image = cv2.imread(img_path)
        
        h, w = image.shape[:2]
        seg_mask = np.full((h, w), 255, dtype=np.uint8) # create the segmentatin mask with all 255 values
        
        # Inference yolov11x-seg and filter out detection that has confidence lower than 0.5
        results = model.predict(img_path, retina_masks=True, show=False, save=True, conf=0.5)
        
        # Loop the results and extract the car masks and insert into the segmentation mask
        for result in results:
            for i, mask in enumerate(result.masks.data):
                cls = result.boxes.cls[i].item() # get the corresponding object ID
                class_name = result.names[int(cls)] # transform the ID into object names (str)
                if class_name == 'car':
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    seg_mask[mask_np > 0] = 0 # apply the mask from the yolo model to the final segmentation mask
        
        # Save the instance segmentation mask
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, sample_name + ".png")
        cv2.imwrite(save_path, seg_mask)
        print(f"Saved instance mask for {sample_name}")
        
if __name__ == '__main__':
    main()
