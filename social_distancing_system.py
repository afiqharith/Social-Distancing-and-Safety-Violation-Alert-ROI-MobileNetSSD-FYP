from config.config import *
import numpy as np
import threading
import math
import cv2
import sys
import os

FONTS = cv2.FONT_HERSHEY_SIMPLEX
VIDEOPATH = os.path.join(os.getcwd(), FOLDERNAME, VIDEONAME)
WEIGHTSPATH = os.path.join(os.getcwd(), MODELPATH, WEIGHTS)
PROTOTXTPATH = os.path.join(os.getcwd(), MODELPATH, PROTOTXT)

class App:
    def __init__(self, VIDEOPATH, CAMERA, START = True):
        if CAMERA == True:
            self.video = cv2.VideoCapture(0)
        else:
            self.video = cv2.VideoCapture(VIDEOPATH)
        self.flag = True
        if START == True: self.main()

    # @param func: (xmin, ymin, xmax, ymax)
    def calculateCentroid(self, *param):
        return (((param[0] + param[2])/2), ((param[1] + param[3])/2))

    # @param func: (xc1, yc1, xc2, yc2)
    def calculateDistance(self, *param):
        # Apply Euclidean distance between two centre points
        return math.sqrt((param[0]-param[2])**2 + (param[1]-param[3])**2)

    def main(self):

        try:
            net = cv2.dnn.readNetFromCaffe(PROTOTXTPATH, WEIGHTSPATH)
        except Exception:
            sys.stdout.write('[FAILED] Unable to load model.')

        while(self.flag):

            detectedBox = []
            centroids = []
            boxColors = []

            self.flag, self.frame = self.video.read()
            if THREAD == True:
                self.thread_1 = threading.Thread(target=self.video.read)
                self.thread_1.daemon = True
                self.thread_1.start()
            else:
                pass

            if self.flag:
                self.frameResized = cv2.resize(self.frame,(300,300))
            else:
                break

            self.frame = cv2.cvtColor(self.frame, cv2.IMREAD_COLOR)

            blob = cv2.dnn.blobFromImage(self.frameResized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
            net.setInput(blob)
            detections = net.forward()

            width = self.frameResized.shape[1] 
            height = self.frameResized.shape[0]

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > THRESHOLD:
                    classID = int(detections[0, 0, i, 1])

                    xmin = int(detections[0, 0, i, 3] * width) 
                    ymin = int(detections[0, 0, i, 4] * height)
                    xmax   = int(detections[0, 0, i, 5] * width)
                    ymax   = int(detections[0, 0, i, 6] * height)

                    heightFactor = self.frame.shape[0]/300.0
                    widthFactor = self.frame.shape[1]/300.0

                    xmin = int(widthFactor * xmin) 
                    ymin = int(heightFactor * ymin)
                    xmax = int(widthFactor * xmax)
                    ymax = int(heightFactor * ymax)

                    if classID != 15:
                        continue    

                    centroid = self.calculateCentroid(xmin,ymin,xmax,ymax)

                    detectedBox.append([xmin,ymin,xmax,ymax,centroid])

                    violation = 0
                    for k in range (len(centroids)):
                        c = centroids[k]
                        if self.calculateDistance(c[0], c[1], centroid[0], centroid[1]) <= DISTANCE:
                            boxColors[k] = True
                            violation = True
                            cv2.line(self.frame, (int(c[0]), int(c[1])), (int(centroid[0]), int(centroid[1])), YELLOW, 1, cv2.LINE_AA)
                            cv2.circle(self.frame, (int(c[0]), int(c[1])), 3, ORANGE, -1,cv2.LINE_AA)
                            cv2.circle(self.frame, (int(centroid[0]), int(centroid[1])), 3, ORANGE, -1, cv2.LINE_AA)
                            break
                    centroids.append(centroid)
                    boxColors.append(violation)

            for i in range (len(detectedBox)):
                x1 = detectedBox[i][0]
                y1 = detectedBox[i][1]
                x2 = detectedBox[i][2]
                y2 = detectedBox[i][3]

                if boxColors[i] == 0:
                    cv2.rectangle(self.frame,(x1,y1),(x2,y2), WHITE, 2,cv2.LINE_AA)
                    label = "safe"
                    labelSize, baseLine = cv2.getTextSize(label, FONTS, 0.5, 2)

                    y1label = max(y1, labelSize[1])
                    cv2.rectangle(self.frame, (x1, y1label - labelSize[1]),(x1 + labelSize[0], y1 + baseLine), (255, 255, 255), cv2.FILLED)
                    cv2.putText(self.frame, label, (x1, y1), FONTS, 0.5, GREEN, 1,cv2.LINE_AA)
                else:
                    cv2.rectangle(self.frame,(x1,y1),(x2,y2), RED, 2)
                    label = "unsafe"
                    labelSize, baseLine = cv2.getTextSize(label, FONTS, 0.5, 2)

                    y1label = max(y1, labelSize[1])
                    cv2.rectangle(self.frame, (x1, y1label - labelSize[1]),(x1 + labelSize[0], y1 + baseLine), (255, 255, 255), cv2.FILLED)
                    cv2.putText(self.frame, label, (x1, y1), FONTS, 0.5, ORANGE, 1,cv2.LINE_AA)

            cv2.namedWindow('Social Distance', cv2.WINDOW_NORMAL)
            cv2.imshow('Social Distance', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    App(VIDEOPATH, CAMERA)
    