import numpy as np
import argparse
import imutils
import time
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import random
import warnings
warnings.filterwarnings('ignore')

# load the model
model = torch.load('model.pt', map_location=torch.device('cpu')) # map_location for CPU usage
model.eval()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())


CLASS_NAMES = ['__background__', 'pedestrian']

def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0
    
    """
    #img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)

    #img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    print(pred_score)

    pred_t = 0

    pred_t_scores = [pred_score.index(x) for x in pred_score if x > confidence]
    if pred_t_scores:
        pred_t = pred_t_scores[-1]

    #pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]

    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class

from PIL import Image


def segment_instance(img, confidence=0.1, rect_th=3, text_size=1, text_th=3):
    """
    segment_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    masks, boxes, pred_cls = get_prediction(img, confidence)
    #img = cv2.imread(img)
    #img = Image.fromarray(img.astype(np.uint8))
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
      #rgb_mask = get_coloured_mask(masks[i])
      #img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
      print(int(boxes[i][0][0]))

      #cv2.rectangle(img, (100, 20), (0, 0), (255, 0, 0), 2)

      #print(pred_cls[i], boxes[i][0])

      cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])), color=(0, 255, 0), thickness=rect_th)
      cv2.putText(img,pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    
    return img

# load the COCO class labels our Mask R-CNN was trained on
#labelsPath = os.path.sep.join([args["mask_rcnn"], "object_detection_classes_coco.txt"])
LABELS = ['__background__', 'pedestrian'] #open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)

# derive the paths to the Mask R-CNN weights and model configuration
# weightsPath = os.path.sep.join([args["mask_rcnn"],
# 	"frozen_inference_graph.pb"])
# configPath = os.path.sep.join([args["mask_rcnn"],
# 	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])
# # load our Mask R-CNN trained on the COCO dataset (90 classes)
# # from disk
# print("[INFO] loading Mask R-CNN from disk...")
# net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(args["input"])
writer = None
# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	total = -1

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    start = time.time()
    end = time.time()
    
    frame = segment_instance(frame, 0.3)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()







