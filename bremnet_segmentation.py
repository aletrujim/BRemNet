"""
BRemNet (Background Removal Network)
Segmentation of human body

Use:
# python3 bremnet_segmentation.py train --dataset=datasets/body --weights=humanbody.h5
# python3 bremnet_segmentation.py segmentation --weights=humanbody.h5 --video=videos/016/016-1.mp4

Using as tampleate Mask R-CNN model 
https://arxiv.org/abs/1703.06870
"""

import os
import sys
import json
import datetime
import numpy as np
import cv2
import imutils
import skimage.draw
import sklearn.metrics as metric
from PIL import Image as PILImage

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import bremnet model
sys.path.append(ROOT_DIR)
from bremnet.config import Config
from bremnet import bremnet_model as modellib, utils

# Path to trained weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, "humanbody.h5")

# Directory to save logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class BodyConfig(Config):
    """Configuration for training with dataset
    """
    NAME = "body"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


class BodyDataset(utils.Dataset):

    def load_body(self, dataset_dir, subset):
        """Load a subset of the dataset
           Subset to load: train or val
        """
        # Add classes. I have only one class to add.
        self.add_class("body", 1, "body")
        
        # Train or validation dataset
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # x and y coordinates of each region (annotations)
        annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotations = list(annotations.values())  
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height = image.shape[0]
            width = image.shape[1]

            self.add_image(
                "body",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "body":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]): # i = i-1
            # Get index of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. 
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image"""
        info = self.image_info[image_id]
        if info["source"] == "body":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model"""
    # Training dataset
    dataset_train = BodyDataset()
    dataset_train.load_body(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BodyDataset()
    dataset_val.load_body(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def body_segmentation(image, mask):
    """Apply body segmentation
    """
    # Make a grayscale copy of the image. 
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        segmentation = np.where(mask, image, 255).astype(np.uint8)
    else:
        segmentation = gray.astype(np.uint8)
        
    return segmentation


def detect_and_body_segmentation(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the body segmentation
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect body
        r = model.detect([image], verbose=1)[0]
        # Body segmentation
        segmentation = body_segmentation(image, r['masks'])
        # Save output
        file_name = args.image + "_segmented.png"
        skimage.io.imsave(file_name, segmentation)

    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vcapture.get(cv2.CAP_PROP_FPS)/2)

        # Define codec and create video writer
        name_video = os.path.basename(video_path).split(".")[0]
        file_name = "videos/"+ name_video +"/"+ name_video +"_segmented.avi"
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 1
        success = True
        while success:
            #print("frame: ", count)

            # Read next image
            success, image = vcapture.read()
            if success:
		# segment half of the frames
                if count % 2 == 0:
                    # OpenCV returns images as BGR, convert to RGB
                    image = image[..., ::-1]
		    # Save frame unsegmented
                    frame_path = "videos/"+ name_video +"/unsegmented/%03d"% count +".png"
                    skimage.io.imsave(frame_path, image)
                    # Detect body
                    r = model.detect([image], verbose=0)[0]
                    # Body segmentation
                    segmentation = body_segmentation(image, r['masks'])
                    # Save segmented image
                    segmented_path = "videos/"+ name_video +"/segmented/%03d"% count +".png"
                    skimage.io.imsave(segmented_path, segmentation)
                    # Green background (Chroma)
                    green_path = "videos/"+ name_video +"/green/%03d"% count +".png"
                    chroma_segmentation(segmented_path, green_path)
                    # Convert RGB to BGR to save image in video
                    segmentation = segmentation[..., ::-1]
                    # Add image to video writer
                    vwriter.write(segmentation)
                    print("saved to", green_path)
                count += 1
        vwriter.release()
    print("Video segmented saved to ", file_name)
    

def chroma_segmentation(segmented_path, green_path):   
    """Convert background to green"""
    # Read segmented image
    img = PILImage.open(segmented_path)
    img = img.convert("RGBA")
    # Get pixels
    pixdata = img.load()
    # Image size
    width, height = img.size

    for y in range(height):
        for x in range(width):
	    # Change color (white to green) 
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (0, 255, 0, 255)
    # Save Chroma image
    img.save(green_path)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train BRemNet to detect body')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'segmentation'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/body/dataset/",
                        help='Directory of the Raices dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the body segmentation effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the body segmentation effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "segmentation":
        assert args.image or args.video,\
               "Provide --image or --video to apply body segmentation"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BodyConfig()
    else:
        class InferenceConfig(BodyConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.BRemNet(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.BRemNet(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
	COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        weights_path = WEIGHTS_PATH # our weights
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=[
            "bremnet_class_logits", "bremnet_bbox_fc",
            "bremnet_bbox", "bremnet_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or segmentation
    if args.command == "train":
        train(model)
    elif args.command == "segmentation":
        detect_and_body_segmentation(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'segmentation'".format(args.command))
