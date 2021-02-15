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
    

    def roiMask(self, frame_rgb):
        
        overlay = cv2.imread(OVERLAY)
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

    def main(self):

        try:
            net = cv2.dnn.readNetFromCaffe(PROTOTXTPATH, WEIGHTSPATH)
        except:
            sys.stdout.write('[FAILED] Unable to load model.')

        while True:
            # Capture frame-by-frame
            self.ret, self.frame = self.video.read()

            if self.ret:
                self.frameResized = cv2.resize(self.frame,(300,300))
            else:
                break
                

            self.frame = cv2.cvtColor(self.frame, cv2.IMREAD_COLOR)
            modiframe, bg, fg, fg_zero, line, mask, mask_invert = self.roiMask(self.frame)
            
            #region bounding for ROI
            cv2.polylines(modiframe, [line], True, yellow, thickness=2)

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
            cv2.imshow('Full', self.frame)
            
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
