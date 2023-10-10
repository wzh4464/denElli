'''
File: ./genPgm.py
Created Date: Tuesday October 10th 2023
Author: Zihan
-----
Last Modified: Tuesday, 10th October 2023 6:31:52 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
2023-10-10        Zihan	make all images in the data folder have the same format
'''

import os
import cv2

# Define the source and destination directories
src_dir = './data/'
dst_image_dir = './data/image/'
dst_pgm_dir = './data/pgm/'

# Create the destination directories if they don't exist
if not os.path.exists(dst_image_dir):
    os.makedirs(dst_image_dir)
if not os.path.exists(dst_pgm_dir):
    os.makedirs(dst_pgm_dir)

# List all files in the source directory
files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# Filter out only the image files
image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Sort the files for consistent numbering
image_files.sort()

# Rename, move, and convert the files
for idx, filename in enumerate(image_files, start=1):
    # Generate new filename with format 001.jpg
    new_filename = f"{idx:03}.jpg"
    
    # Load the image using OpenCV
    img = cv2.imread(os.path.join(src_dir, filename))
    
    # Save the image to the new location with .jpg format
    cv2.imwrite(os.path.join(dst_image_dir, new_filename), img)
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert the grayscale image to PGM format and save in the pgm folder
    pgm_filename = os.path.join(dst_pgm_dir, f"{idx:03}.pgm")
    cv2.imwrite(pgm_filename, gray_img)

print("Processing complete!")
