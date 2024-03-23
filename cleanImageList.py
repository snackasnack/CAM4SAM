import os 

file_path = "C:/Users/snack/Desktop/CAM4SAM/temp_files/allClassesSegmented.txt"
updated_lines = []

with open(file_path, "r") as file:
    for line in file:
        line = line.strip()
        file_name = line.replace(".npy", "")
        #print(file_name)
        
        try:
            # If image does not have its segmented form
            image_path = os.path.join('C:/Users/snack/Desktop/SEAM/VOC2012_data/SegmentationClass/', file_name + '.png')
            with open(image_path, "rb"):
                pass 
            # If image does not have its .npy form
            image_path = os.path.join('C:/Users/snack/Desktop/SEAM/voc12/out_cam/', file_name + '.npy')
            with open(image_path, "rb"):
                pass 
        except FileNotFoundError:
            print(f"Removing {file_name} from the list")
        else:
            updated_lines.append(file_name)

with open(file_path, "w") as file:
    for line in updated_lines:
        file.write(line + "\n")
