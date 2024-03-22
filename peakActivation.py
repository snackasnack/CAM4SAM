import os 

def read_results():
    None


if __name__ == "__main__":

    directory_path = 'C:/Users/snack/Desktop/CAM4SAM/temp_files/Saved_Results/mask_results'
    files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    for file in files:
        file_path = os.path.join(directory_path, file)
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 3: 
                    iou_result, cam_threshold, sobel_threshold = parts
                    print(f"IoU Result: {iou_result}, CAM Threshold: {cam_threshold}, Sobel Threshold: {sobel_threshold}")
