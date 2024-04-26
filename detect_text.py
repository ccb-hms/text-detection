# code adapted from https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
# Other implementations:
# 1. https://github.com/YCICI/EAST-OpenCv/tree/master
# Some of the example images have been borrowed from:
# 1. Validation set of Text Detection Dataset from 
# https://towardsdatascience.com/object-detection-in-6-steps-using-detectron2-705b92575578
# 2. Creative Common Tagged Images downloaded from Google Images
# EAST paper: https://arxiv.org/abs/1704.03155


import os
import sys
import time
import cv2
import numpy as np
import argparse
from dotenv import load_dotenv
from imutils.object_detection import non_max_suppression


def find_images(io_path):
    img_list =[]
    extension = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    file_list = os.listdir(io_path)

    for files in file_list:
        for ext in extension:
            if files.endswith(ext):
                img_list.append(files)
    
    return img_list


def text_detection(img_path, weights, min_confidence):

    # load the input image and grab the image dimensions
    image = cv2.imread(img_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height. Resized image height & width should 
    # be a multiple of 32
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

    # load the pre-trained weights for EAST text detector
    net = cv2.dnn.readNet(weights)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start_time = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end_time = time.time()
    total_time = end_time - start_time
    
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
	    # data used to derive potential bounding box coordinates that
	    # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if  score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
		    # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
		    # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
		    # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
		    # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
		    # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
	    # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)

    return orig, total_time


def text_detection_command():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--min_confidence", type=float, default=0.5,
        help="Minimum probability required to inspect a region")
    ap.add_argument("-input"  , "--input_path" , type=str, 
        default=os.path.join(os.getcwd(), 'sample_images_text'), 
        help="Image directory. Please specify the path to input images. ")
    ap.add_argument ("-output", "--output_path", type=str, 
        default=os.path.join(os.getcwd(), 'output'), 
        help="Output directory. Please specify the path to output directory where processed images will be saved.")
    ap.add_argument("-weights", "--pretrained_weights", type=str, 
        default=os.path.join(os.getcwd(), 'frozen_weights', 'frozen_east_text_detection.pb'), 
        help="Path to frozen weights for inference. ")
    
    args = vars(ap.parse_args())

    io_path = args["input_path"]
    op_path = args["output_path"]
    
    if not os.path.exists(op_path): 
        os.makedirs(op_path)

    min_confidence = args["min_confidence"]
    pretrained_weights_path = args["pretrained_weights"]
    
    return min_confidence, io_path, op_path, pretrained_weights_path

   
if __name__ == '__main__':
    min_confidence, input_path, output_path, pretrained_weights = text_detection_command() 

    image_list = find_images(input_path)
    duration = 0

    for each_image in image_list:
        image_path = os.path.join(input_path, each_image)
        image_name, extension = each_image.split('.')[0], each_image.split('.')[1]
        image_res = image_name + '_res.' + extension
        image_output = os.path.join(output_path, image_res)

        res, timetaken = text_detection(image_path, pretrained_weights, min_confidence)
        #print ('Image took {:.6f} seconds'.format(timetaken))
        
        duration = duration + timetaken
        cv2.imwrite(image_output, res)
    
    print("[INFO] text detection on {0} images took {1:.6f} seconds".format(len(image_list), duration))
    




    