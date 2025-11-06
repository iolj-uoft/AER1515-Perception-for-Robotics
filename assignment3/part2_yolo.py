import numpy as np
import time
import cv2
import os
import argparse
import matplotlib.pyplot as plt

###########################################################
# OPTIONS
###########################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["train", "test"], default="train", required=True, type=str)
args = parser.parse_args()

if (args.dataset == "train"):
    image_dir = 'data/train/left'
    sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']
else:
    image_dir = 'data/test/left'
    sample_list = ['000011', '000012', '000013', '000014', '000015']

img_list = []
name_list = []

yolo_dir = 'yolo'

# minimum probability to filter weak detections
confidence_th = 0.51

# threshold when applyong non-maxima suppression
threshold = 0.6
###########################################################

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configurationY
weightsPath = os.path.sep.join([yolo_dir, "yolov3.weights"])
configPath = os.path.sep.join([yolo_dir, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

for sample_name in sample_list:
    # load our input image and grab its spatial dimensions
    img_path = os.path.join(image_dir, sample_name + ".png")
    image = cv2.imread(img_path)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_th:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_th,
                            threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    img_list.append(image)
    name_list.append(sample_name)
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)

if (args.dataset == "test"):
    if len(img_list) > 0:
        n = len(img_list)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 3 * n))
        # When n==1, axes is not an array
        if n == 1:
            axes = [axes]
        for i, (img, name) in enumerate(zip(img_list, name_list)):
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"{name}")
            axes[i].axis('off')
            axes[i].set_title(f"Sample: {name}")
        plt.tight_layout()
        out_path = os.path.join('figures', 'yolo_detections_stack.png')
        os.makedirs('figures', exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved subplot to {out_path}")
