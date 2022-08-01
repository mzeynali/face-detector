import os
import cv2
import numpy as np
import mediapipe as mp
from .analysis import FaceMesh

def create_model_path():
    model_path = os.path.join(os.path.expanduser('~'),'.weights','canonical_face_model.obj')
    if not os.path.isfile(model_path):
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        FACE_MODEL_URL = 'https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj'
        import requests
        r = requests.get(FACE_MODEL_URL, allow_redirects=True)
        with open(model_path, 'wb') as f :
            f.write(r.content)
    return model_path
        
class FaceDetector:
    
    def __init__(self,use_facemesh=True,model_path=None):
        self.use_facemesh = use_facemesh
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)
        if model_path is None:
            model_path = create_model_path()
        if use_facemesh:
            self.face_mesh = FaceMesh(facemodel_path=model_path)
            
    def __call__(self, bgr_image):
        if self.use_facemesh:
            faces_crop, bboxes, keypoints, rotate_degree, eyes_ratio, eyes_center = self.by_facemesh(bgr_image)
        else:
            faces_crop, bboxes, keypoints, rotate_degree, eyes_ratio, eyes_center = self.by_blazeface(bgr_image)         
        return faces_crop, bboxes, keypoints, rotate_degree, eyes_ratio, eyes_center

    def by_facemesh(self, bgr_image):
        aligned_face, real_bbox, keypoints, rotate_degree, eyes_ratio, eyes_center = self.face_mesh(bgr_image)
        if aligned_face is not None:
            bboxes = real_bbox.reshape(-1,4)
            faces_crop = [aligned_face]
        else:
            faces_crop, bboxes, keypoints, rotate_degree, eyes_ratio, eyes_center = None,None,None,None,(None,None),(None,None)
        return faces_crop, bboxes, keypoints, rotate_degree, eyes_ratio, eyes_center
    
    def by_blazeface(self, bgr_image):
        results = self.detector.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        bboxes = results.detections
        if bboxes is not None:
            imH,imW = bgr_image.shape[:2]
            bboxes = [(bbx.location_data.relative_bounding_box.xmin, bbx.location_data.relative_bounding_box.ymin,
                       bbx.location_data.relative_bounding_box.xmin+bbx.location_data.relative_bounding_box.width,
                       bbx.location_data.relative_bounding_box.ymin+bbx.location_data.relative_bounding_box.height)  for bbx in bboxes]
            bboxes = (np.array(bboxes) * np.array([imW,imH,imW,imH])).astype(int)
            faces_crop = [bgr_image[bbox[1]:bbox[3],bbox[0]:bbox[2]] for bbox in bboxes]
            keypoints,rotate_degree,eyes_ratio = None,None,(None,None)
        else:
            faces_crop, bboxes, keypoints, rotate_degree, eyes_ratio, eyes_center = None,None,None,None,(None,None),(None,None) 
        return faces_crop, bboxes, keypoints, rotate_degree, eyes_ratio, eyes_center

