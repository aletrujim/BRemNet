'''
frames2video.py

script to convert frames to video format .avi
Detection and segmentation of body in video

by: @aletrujim
'''

import os
import cv2
import argparse

# Parse arguments
parse = argparse.ArgumentParser(description = 'frames2video: Convert frames to video')
parse.add_argument('--directory', type = str, metavar = 'path', default ='mask',
	help = 'Directory for where the images are stored')
parse.add_argument('--name', type = str, metavar = 'name', default ='new_video',
	help = 'name of the video file')
parse.add_argument('--fps', type = int, metavar = 'fps', default = 2,
    help = 'frames per second')
args = parse.parse_args()

# Placeholding the arguments and ensuring that the format is of .avi
image_folder = args.directory
video_name = args.name
video_name = video_name + '.avi'
fps = args.fps

# Load all the frames in the folder, only if it ends with .jpg 
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Declaring OpenCV's VideoWriter function
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))

# Sorts the images to ensure they are ordered
images.sort()

# Concatenate all the images to the video format
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Close and release the video file. 
cv2.destroyAllWindows()
video.release()
