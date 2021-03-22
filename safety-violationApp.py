import cv2
import numpy as np
import math
import os
import sys
from utils.config import *
from utils import thread

FONTS = cv2.FONT_HERSHEY_SIMPLEX
VIDEOPATH = os.path.join(os.getcwd(), FOLDERNAME, VIDEONAMEROI)
WEIGHTSPATH = os.path.join(os.getcwd(), MODELPATH, WEIGHTS)
PROTOTXTPATH = os.path.join(os.getcwd(), MODELPATH, PROTOTXT)
OVERLAY = os.path.join(os.getcwd(), IMAGESFOLDER, OVERLAYNAME)

class AppROI:

    def __init__(self, VIDEOPATH, CAMERA, START = True):

        if CAMERA == True and THREAD == True:
            self.video = thread.ThreadingClass(0)
        elif CAMERA == True and THREAD == False:
            self.video = cv2.VideoCapture(0)
        elif CAMERA == False and THREAD == True:
            self.video = thread.ThreadingClass(VIDEOPATH)
        else:
            self.video = cv2.VideoCapture(VIDEOPATH)

        if START == True:
            self.main()
    
    def roiMask(self, *src):
        overlay = cv2.imread(OVERLAY)
        overlay = cv2.resize(overlay, src[0].shape[1::-1])
        opac = cv2.addWeighted(src[0], 0.5, overlay, 0.3,0)
        mask = np.zeros(src[0].shape[0:2], dtype=np.uint8)
        point = np.array([[190,230],[490,230],[450,90],[250,90]])
        line = np.array([[190,230],[490,230],[450,90],[250,90]])

        # draw mask
        cv2.fillConvexPoly(mask, point, (255, 255, 255))

        # get first masked value (foreground) [fg,fg]
        fg = cv2.bitwise_or(opac, opac, mask=mask)
        fg_zero = cv2.bitwise_or(opac, opac, mask=mask) #fg with opac without line

        # get second masked value (background) mask must be inverted
        mask_invert = cv2.bitwise_not(mask) #untuk dapatkan mask foreground
        background = np.full(src[0].shape, src[0], dtype=np.uint8)
        bg = cv2.bitwise_or(background, background, mask=mask_invert)

        # combine foreground+background (combining fg and background w bitwise or)
        modiframe = cv2.bitwise_or(fg, bg)
        zero = cv2.bitwise_or(fg, bg)
        return modiframe, line
    
    def check_location(self, *args):
        if 190 <= args[0] <= 490 or 190 <= args[1] <= 490:
            if 90 <= args[2] <= 230:
                return True
            else:
                return False
        return False

    def main(self):
        try:
            net = cv2.dnn.readNetFromCaffe(PROTOTXTPATH, WEIGHTSPATH)
        except Exception:
            sys.stdout.write('[FAILED] Unable to load model.')

        while(self.video.isOpened()):
            # Capture frame-by-frame
            self.flag, self.frame = self.video.read()

            if self.flag:
                self.frameResized = cv2.resize(self.frame,(300,300), interpolation = cv2.INTER_AREA)
            else:
                break
                
            self.frame = cv2.cvtColor(self.frame, cv2.IMREAD_COLOR)
            self.frame_modified, self.line = self.roiMask(self.frame)
            
            #region bounding for ROI
            cv2.polylines(self.frame_modified, [self.line], True, YELLOW, thickness=2)

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
                        
                    if self.check_location(xmin, xmax, ymax) == True:
                        cv2.rectangle(self.frame_modified, (xmin,ymin), (xmax,ymax), RED, 2)
                        label = 'err_area'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                        yminlabel = max(ymin, labelSize[1])
                        cv2.rectangle(self.frame_modified, (xmin, yminlabel - labelSize[1]),(xmin + labelSize[0], ymin + baseLine), RED, cv2.FILLED)
                        cv2.putText(self.frame_modified, label, (xmin, yminlabel),cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1,cv2.LINE_AA)

            #Display output
            cv2.imshow('ROI', self.frame_modified)
            
            # Break with ESC 
            if cv2.waitKey(1) >= 0:  
                break

        if THREAD == True:
            pass
        else:
            self.video.release()

if __name__ == '__main__':
    AppROI(VIDEOPATH, CAMERA)
    cv2.destroyAllWindows()
