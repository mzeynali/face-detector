import mediapipe as mp
import cv2
import numpy as np
import math
from collections import namedtuple

class FaceMesh:
    def __init__(self, facemodel_path = './canonical_face_model.obj'):
        self.points = []
        Point = namedtuple('Point', 'x y z')
        with open(facemodel_path) as f:
            for line in f:
                line = line[:-1].split(' ')
                if line[0] == 'v':
                    x, y, z = map(float, line[1:])
                    self.points.append(Point(x, y, z))
                    
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                            max_num_faces=1,
                            refine_landmarks=True, 
                            min_detection_confidence=0.2,
                            min_tracking_confidence=0.5)
        
    def analysis(self, frame):
        rotate_degree, keypoints = None, None
        frame.flags.writeable = False
        results = self.face_mesh.process(frame)
        frame.flags.writeable = True
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            keypoints = []
            imH,imW = frame.shape[:2]
            for data_point in face_landmarks.landmark:
                keypoints.append({'X': data_point.x*imW,
                                  'Y': data_point.y*imH })
            imgpts, modelpts, rotate_degree, nose = center_orientation(frame,keypoints,self.points)
        return rotate_degree, keypoints

    def __call__(self, frame):
        imH,imW = frame.shape[:2]
        rotate_degree, keypoints =  self.analysis(frame)
        
        if keypoints is not None:
            kpts_array = np.zeros((len(keypoints),2),dtype=int)
            for ki, kpt in enumerate(keypoints):
                x,y = int(kpt.get('X')), int(kpt.get('Y'))
                kpts_array[ki] = [x,y]
            real_xmin,real_ymin = kpts_array.min(0)
            real_xmax,real_ymax = kpts_array.max(0)
            real_bbox = np.array([real_xmin,real_ymin,real_xmax,real_ymax])
            rot_mat = cv2.getRotationMatrix2D((imW//2,imH//2), -rotate_degree[0], 1.0)
            ones = np.ones(shape=(len(kpts_array), 1))
            transformed_points = np.hstack([kpts_array, ones])
            transformed_points = rot_mat.dot(transformed_points.T).T.astype(int)
            xmin,ymin = transformed_points.min(0)
            xmax,ymax = transformed_points.max(0)
            warped_frame = cv2.warpAffine(frame, rot_mat, (imW,imH), flags=cv2.INTER_LINEAR)
            aligned_face = warped_frame[ymin:ymax,xmin:xmax]  
            
            kpts_array = kpt2array(keypoints)
            rcenter, lcenter, right_eye_kpts, left_eye_kpts = get_eye_kpts(kpts_array)
            r_ratio, l_ratio  = get_eye_ratio(right_eye_kpts, left_eye_kpts)
            eyes_ratio  = (r_ratio, l_ratio)
            eyes_center = (rcenter, lcenter)
            keypoints = kpts_array
        else:
            aligned_face, real_bbox, keypoints, rotate_degree, eyes_ratio, eyes_center = None, None, None, None, (None, None), (None, None)
            
        return aligned_face, real_bbox, keypoints, rotate_degree, eyes_ratio, eyes_center

def kpt2array(keypoints):
    kpts_array = np.zeros((len(keypoints),2))
    for k,kpt in enumerate(keypoints):
        kpts_array[k] = [kpt['X'],kpt['Y']]
    return kpts_array.astype(int)

def get_eye_kpts(kpts_array):
    right_eye_kpt_idxs = {
        'outer':[130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25],
        'inner':[33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,]
        }
    
    left_eye_kpt_idxs = {
        'outer':[359,467,260,259,257,258,286,414,463,341,256,252,253,254,339,255],
        'inner':[263,466,388,387,386,385,384,398,382,382,381,380,374,373,390,249]
        }  

    right_eye = kpts_array[right_eye_kpt_idxs['inner']]
    rcenter  = kpts_array[-10]
    left_eye = kpts_array[left_eye_kpt_idxs['inner']]
    lcenter = kpts_array[-5]
    
    return rcenter, lcenter, right_eye, left_eye

def get_eye_ratio(right_eye_kpts, left_eye_kpts):
    r_left,r_right,r_up,r_down = right_eye_kpts[[0,8,4,12]]
    l_left,l_right,l_up,l_down = left_eye_kpts[[0,8,4,12]]
    r_w = np.linalg.norm(r_left-r_right)
    r_h = np.linalg.norm(r_up-r_down)
    l_w = np.linalg.norm(l_left-l_right)
    l_h = np.linalg.norm(l_up-l_down)    
    r_ratio, l_ratio = r_h/r_w, l_h/l_w
    return   r_ratio, l_ratio  


def center_orientation(frame,keypoints,points,num=40):
        size = frame.shape #(height, width, color_channel)
        center_point = np.array([size[1]/2, size[0]/2])
        indexes = [6,168,193,417, # nose
                    33,263,133,362, # eye
                    54,284,10,109,338,67,297,162,389,103,332, # top of head
                    342,353,383,368,389,
                    113,124,156,139,162,
                    226,35,143,34,127,
                    446,265,372,264,356,
                    68,104,69,108,151,337,299,333,298,
                    25,31,111,116,227,234,
                    255,261,340,345,447,454,
                    63,105,66,107,9,336,296,334,293,
                    ] 
        image_points = np.array([(keypoints[ii]['X'], keypoints[ii]['Y']) for ii in indexes], dtype="double")
        image_points = image_points - image_points[0] + center_point
        model_points = np.array([(points[jj][0]*num, points[jj][1]*num, points[jj][2]*num) for jj in indexes])
        model_points = model_points - model_points[0]
        
        # Camera internals
        center = (size[1]/2, size[0]/2)
        focal_length = center[0] / np.tan(60/2 * np.pi / 180)
        camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
        axis = np.float32([[300,0,0], 
                              [0,300,0], 
                              [0,0,300]])
        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 
    
        
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
    
    
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
    
        return imgpts, modelpts, (roll, pitch, yaw), image_points[0].astype(int)
