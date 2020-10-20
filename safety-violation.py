import cv2
import numpy as np
import argparse
import time

#[b,g,r]
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
yellow = (0,255,255)


# construct the argument parse 
parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network ')
parser.add_argument('--video', help='path to video file. If empty, camera stream will be used', 
                    default="PRG23.mp4" )
parser.add_argument('--prototxt', help='Path to text network file [filename.prototxt] ', 
                    default='utils/MobileNetSSD_deploy.prototxt')
parser.add_argument('--weights', help='Path to weights [filename.caffemodel] ',
                    default='utils/MobileNetSSD_deploy.caffemodel')
parser.add_argument("--threshold", help='confidence threshold to filter out weak detections',
                    default=0.2, type=float)
args = parser.parse_args()

def roiMask(frame_rgb):

    overlay = cv2.imread('overlay.jpg')
    overlay = cv2.resize(overlay, frame_rgb.shape[1::-1])
    opac = cv2.addWeighted(frame_rgb,0.5,overlay,0.3,0)

    mask = np.zeros(frame_rgb.shape[0:2], dtype=np.uint8)

    point = np.array([[190,230],[490,230],[450,90],[250,90]])
    line = np.array([[190,230],[490,230],[450,90],[250,90]])

    # draw mask
    cv2.fillConvexPoly(mask, point, (255, 255, 255))

    # get first masked value (foreground) [fg,fg]
    fg = cv2.bitwise_or(opac, opac, mask=mask)
    fg_zero = cv2.bitwise_or(opac, opac, mask=mask) #fg with opac without line

    # get second masked value (background) mask must be inverted
    mask_invert = cv2.bitwise_not(mask) #untuk dapatkan mask foreground
    background = np.full(frame_rgb.shape, frame_rgb, dtype=np.uint8)
    bg = cv2.bitwise_or(background, background, mask=mask_invert)

    # combine foreground+background (combining fg and background w bitwise or)
    modiframe = cv2.bitwise_or(fg, bg)
    zero = cv2.bitwise_or(fg, bg)
    return modiframe, bg, fg, fg_zero, line, mask, mask_invert


#Load the Caffe model 
print("[INFO] Load Caffe model detection..")
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

print("[INFO] Opening video..")
video = cv2.VideoCapture(args.video)

while True:

    # Capture frame-by-frame
    ret, frame = video.read()

    if ret:
        frame_resized = cv2.resize(frame,(300,300),interpolation = cv2.INTER_AREA) # resize frame for prediction
        # frame_resized = imutils.resize(frame, width=int(300))
        

    frame_rgb = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

    #masking
    modiframe, bg, fg, fg_zero, line, mask, mask_invert = roiMask(frame_rgb)
    
    #region bounding for ROI
    cv2.polylines(modiframe, [line], True, yellow, thickness=2)

    # MobileNet requires fixed dimensions for input image(s)
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    #Set to network the input blob 
    net.setInput(blob)
    #Prediction of network
    detections = net.forward()

    #Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    #For get the class and location of object detected, 
    # There is a fix index for class, location and confidence
    # value in @detections array .
    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > args.threshold: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label
            
            
            # Object location 
            xmin = int(detections[0, 0, i, 3] * cols) 
            ymin = int(detections[0, 0, i, 4] * rows)
            xmax   = int(detections[0, 0, i, 5] * cols)
            ymax   = int(detections[0, 0, i, 6] * rows)
            
            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0
            widthFactor = frame.shape[1]/300.0

            # Scale object detection to frame
            xmin = int(widthFactor * xmin) 
            ymin = int(heightFactor * ymin)
            xmax   = int(widthFactor * xmax)
            ymax   = int(heightFactor * ymax)

            # if class id is not 15(person), ignore it
            if class_id != 15:
                continue
                
            if 190 <= xmin <= 490 or 190 <= xmax <= 490:
                if 90 <= ymax <= 230:
                    cv2.rectangle(modiframe, (xmin,ymin), (xmax,ymax), red, 2)
                    label = 'Trespass'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    yminlabel = max(ymin, labelSize[1])
                    cv2.rectangle(frame_rgb, (xmin, yminlabel - labelSize[1]),(xmin + labelSize[0], y1 + baseLine), (255, 255, 255), cv2.FILLED)
                    cv2.putText(modiframe, label, (xmin, yminlabel-4),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2,cv2.LINE_AA)
            
                else:
                    pass
        
            else:
                pass

    #Display output
    cv2.imshow('Full', modiframe)
    
    # Break with ESC 
    if cv2.waitKey(1) >= 0:  
        break


video.release()
cv2.destroyAllWindows()
