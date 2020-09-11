import cv2
import numpy as np
import argparse
import math
import config

#[b,g,r]
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
yellow = (0,255,255)
white = (255,255,255)
orange = (0,165,255)
font = cv2.FONT_HERSHEY_SIMPLEX

# construct the argument parse 
parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network ')
parser.add_argument('--video', help='path to video file. If empty, camera stream will be used', 
                    default="TownCentre.mp4" )
parser.add_argument('--prototxt', help='Path to text network file [filename.prototxt] ', 
                    default='MobileNetSSD_deploy.prototxt')
parser.add_argument('--weights', help='Path to weights [filename.caffemodel] ',
                    default='MobileNetSSD_deploy.caffemodel')

args = parser.parse_args()

if args.video == 'TownCentre.mp4':
    threshold = config.towncentre['threshold']
    distance = config.towncentre['distance']

if args.video == 'PETS2009.mp4':
    threshold = config.pets2009['threshold']
    distance = config.pets2009['distance']

if args.video == 'VIRAT.mp4':
    threshold = config.virat['threshold']
    distance = config.virat['distance']

def calculateCentroid(xmin,ymin,xmax,ymax):

    xmid = ((xmax+xmin)/2)
    ymid = ((ymax+ymin)/2)
    centroid = (xmid,ymid)

    return xmid,ymid,centroid


def get_distance(x1,x2,y1,y2):
  distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
  return distance

#Load the Caffe model 
print("[INFO] Load Caffe model detection..")
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

print("[INFO] Opening video..")
#video = instanceThread().start()
video = cv2.VideoCapture(args.video)

while True:

    # store bounding boxes points
    detectedBox = []

    # Capture frame-by-frame
    #ret, frame = video.ret, video.frame
    ret, frame = video.read()    
    
    if ret:
        frame_resized = cv2.resize(frame,(300,300),interpolation=cv2.INTER_AREA) # resize frame for prediction       
    else:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

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
    centroids = []
    box_colors = []
    for i in range(detections.shape[2]):
        
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > threshold: # Filter prediction 
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

            # if class id is 15(person)
            if class_id == 15:               
                
                #calculate centroid point for bounding boxes
                xmid, ymid, centroid = calculateCentroid(xmin,ymin,xmax,ymax)

                #subtitute xmin,ymin,xmax,ymax,centroid into array
                detectedBox.append([xmin,ymin,xmax,ymax,centroid])

                my_color = 0
                for k in range (len(centroids)):
                  c = centroids[k]
                  if get_distance(c[0],centroid[0],c[1],centroid[1]) <= distance:
                    box_colors[k] = 1
                    my_color = 1
                    cv2.line(frame_rgb, (int(c[0]),int(c[1])), (int(centroid[0]),int(centroid[1])), yellow, 1,cv2.LINE_AA)
                    cv2.circle(frame_rgb, (int(c[0]),int(c[1])), 3, orange, -1,cv2.LINE_AA)
                    cv2.circle(frame_rgb, (int(centroid[0]),int(centroid[1])), 3, orange, -1,cv2.LINE_AA)
                    break
                
                centroids.append(centroid)
                box_colors.append(my_color) # 0 or 1                
    
    for i in range (len(detectedBox)):
      x1 = detectedBox[i][0]
      y1 = detectedBox[i][1]
      x2 = detectedBox[i][2]
      y2 = detectedBox[i][3]
      
      if box_colors[i] == 0:
          cv2.rectangle(frame_rgb,(x1,y1),(x2,y2), white, 2,cv2.LINE_AA)
          label = "safe"
          labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

          y1label = max(y1, labelSize[1])
          cv2.rectangle(frame_rgb, (x1, y1label - labelSize[1]),(x1 + labelSize[0], y1 + baseLine), (255, 255, 255), cv2.FILLED)
          cv2.putText(frame_rgb, label, (x1, y1), font, 0.5, green, 1,cv2.LINE_AA)
      else:
          cv2.rectangle(frame_rgb,(x1,y1),(x2,y2), red, 2)
          # cv2.ellipse(frame, centroide, (35, 19), 0.0, 0.0, 360.0, red, 2)

          label = "unsafe"
          labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

          y1label = max(y1, labelSize[1])
          cv2.rectangle(frame_rgb, (x1, y1label - labelSize[1]),(x1 + labelSize[0], y1 + baseLine), (255, 255, 255), cv2.FILLED)
          cv2.putText(frame_rgb, label, (x1, y1), font, 0.5, orange, 1,cv2.LINE_AA)
  

    #display output
    cv2.imshow('Social Distance', frame_rgb)
    
    # Break with ESC 
    if cv2.waitKey(1) >= 0:  
        break

cv2.destroyAllWindows()
video.release()

