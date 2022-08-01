import cv2
import numpy as np
from face_detector import FaceDetector

if __name__ == '__main__':
    fdet = FaceDetector(use_facemesh=True)
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        if not ret :
            break
        canvas = frame.copy()
        faces, bboxes, kpts, angles, ratios = fdet(frame)
        if bboxes is not None:
            for bbx in bboxes:
                xmin,ymin,xmax,ymax = bbx
                cv2.rectangle(canvas,(xmin,ymin),(xmax,ymax),(255,0,0),2)
        
        cv2.imshow('r',canvas)
        k = cv2.waitKey(2)
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    
