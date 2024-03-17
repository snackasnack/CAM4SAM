import os 

file_path = "C:/Users/snack/Desktop/CAM4SAM/temp_files/classZeroSingleInstanceImages.txt"
updated_lines = []

with open(file_path, "r") as file:
    for line in file:
        line = line.strip()
        file_name = line.replace(".npy", "")
        print(file_name)
        
        try:
            # Attempt to read the file path
            image_path = os.path.join('C:/Users/snack/Desktop/SEAM/VOC2012_data/SegmentationClass/', file_name + '.png')
            with open(image_path, "rb"):
                pass  # If successful, do nothing
        except FileNotFoundError:
            # If the file is not found, remove the corresponding file name from the list
            print(f"Removing {file_name} from the list")
        else:
            # If no exception occurred, keep the line in the updated list
            updated_lines.append(line)

# Write back the updated lines to the file
with open(file_path, "w") as file:
    for line in updated_lines:
        file.write(line + "\n")
