from typing import Optional

import datetime
import logging
import pathlib

import cv2
import numpy as np
import yacs.config

from ptgaze import (Face, FacePartsName, GazeEstimationMethod, Visualizer)#GazeEstimator,
import gaze_estimator
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: yacs.config.CfgNode,timeList,coordinate,scalePercentage,fps,startList,endList): ## Snigdha parameter added -- timeList,coordinate,fps
        self.config = config
        self.gaze_estimator = gaze_estimator.GazeEstimator(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        # self.writer = self._create_video_writer() ## Snigdha commented
        self.text_writer = self._create_text_writer()

        self.videoTimeList = timeList ## Snigdha added
        self.coordinate = coordinate ## Snigdha added
        self.fps = fps ## Snigdha added
        self.writeData = '' ##Snigdha added
        self.scalePercentage = scalePercentage ##Snigdha added
        self.startList = startList ##Snigdha added
        self.endList = endList ##Snigdha added

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError

    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)
        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                # cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            # cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    def _run_on_video(self) -> None:
        index = -1
        frameNo = self.startList[0] # Snigdha added 
        # print(self.startList,self.endList)
        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break
            
            ok, frame = self.cap.read()
            
            if not ok:
                break
            # Snigdha added started
            # if frameNo <= 30*5: ## Soumyajit Attention
            # if frameNo <= 60*1 or frameNo >= 60*19:
            if self.videoTimeList:
                if not any(start<=frameNo/self.fps<=end for start,end in self.videoTimeList):
                    continue
            if index == -1:
                index = 0
                self.cap.set(1,self.startList[index])
            elif frameNo > self.endList[index] and index+1 < len(self.startList):
                # print(frameNo)
                index += 1
                frameNo = self.startList[index]
                self.cap.set(1,self.startList[index])
            elif frameNo > self.endList[index] and index+1 == len(self.startList):
                break
            # print(frameNo,index)
            # else:
            #     print(frameNo,frameNo/self.fps)
            # Snigdha added ended
            # startTime = datetime.datetime.now()
            self._process_image(frame,frameNo) ## Snigdha parameter added -- frameNo
            frameNo += 1 # Snigdha added
            # endTime =datetime.datetime.now()
            # print(endTime-startTime)
            # if self.config.demo.display_on_screen: ## Snigdha commented
            #     cv2.imshow('frame', self.visualizer.image) ## Snigdha commented
        self.cap.release()
        # if self.writer:
        #     self.writer.release()
        if self.text_writer:
            self.text_writer.write(self.writeData)
            self.text_writer.close()

    def _process_image(self, image,frameNo) -> None: ## Snigdha parameter added -- frameNo
        rawDim = (image.shape[1],image.shape[0])
        x = self.coordinate[0]
        y = self.coordinate[1]
        h = self.coordinate[2]
        w = self.coordinate[3]
        image = image[x:w,y:h]
        # scale_percent = 40 # percent of original size
        width = int(image.shape[1] * self.scalePercentage / 100)
        height = int(image.shape[0] * self.scalePercentage / 100)
        dim = (width, height)
          
        # resize image
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)
        # # Snigdha added start
        # ## Soumyajit Attention
        # x = 0
        # y = 0
        # h = 600
        # w = 450
        ## TeachingTool
        # x = 0
        # y = 0
        # h = 600
        # w = 330
        
        # # Snigdha added end
        # self.visualizer.set_image(image.copy()) # Snigdha commented
        # self.visualizer.set_image(image.copy()) # Snigdha added
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            # print(frameNo)
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face,frameNo)
            self._draw_head_pose(face,frameNo)
            self._draw_landmarks(face,frameNo,rawDim,image)
            # self._draw_face_template_model(face)
            self._draw_gaze_vector(face,frameNo)
            # self._display_normalized_image(face) ## Snigdha commented

        # if self.config.demo.use_camera:
        #     self.visualizer.image = self.visualizer.image[:, ::-1]
        # if self.writer: ## Snigdha commented
        #     self.writer.write(self.visualizer.image) ## Snigdha commented
        # if self.text_writer: ## Snigdha commented
        #     self.text_writer.write(self.eye.gaze_vector)#self.eye.gaze_vector ## Snigdha commented

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')#H264
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        # if writer is None:
        #     raise RuntimeError
        return writer

    def _create_text_writer(self):
        
        ext = 'txt'
        
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        text_writer = open(output_path,'w')
        # if writer is None:
        #     raise RuntimeError
        return text_writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face,frameNo) -> None:
        if not self.show_bbox:
            return
        # self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face,frameNo) -> None:
        # if not self.show_head_pose:
        #     return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        # self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        # print(euler_angles)
        self.writeData += str(round(frameNo/self.fps,2))+',head,'+str(round(pitch,2))+','+str(round(yaw,2))+','+str(round(roll,2))+'\n'#+','.join(str(x) for x in [pitch, yaw, roll])+'\n' ## Snigdha Added
        # logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
        #             f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face,frameNo,dim,image) -> None:
        # print(face.landmarks)
        # print(dim)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self._pointProjects(face.landmarks,dim,frameNo)
        # print(face.landmarks[36])
        left_side_white_ratio_left_eye,right_side_white_ratio_left_eye = self._gaze_ratio([36,37,38,39,40,41],face.landmarks,gray)
        left_side_white_ratio_right_eye,right_side_white_ratio_right_eye = self._gaze_ratio([42,43,44,45,46,47],face.landmarks,gray)
        left_side = (left_side_white_ratio_left_eye+left_side_white_ratio_right_eye)/2
        right_side = (right_side_white_ratio_left_eye+right_side_white_ratio_right_eye)/2
        # print(frameNo,round(left_side,2),round(right_side,2))
        eyeball_pointing = 'N'
        if left_side > right_side: 
            # print(frameNo,'R')
            eyeball_pointing = 'R'
        else:
            # print(frameNo,'L') 
            eyeball_pointing = 'L'
        self.writeData += str(round(frameNo/self.fps,2))+',GR,'+ str(eyeball_pointing)+'\n'

        # gaze_ratio_left_eye = self._gaze_ratio([36,37,38,39,40,41],face.landmarks,gray)
        # gaze_ratio_right_eye = self._gaze_ratio([42,43,44,45,46,47],face.landmarks,gray)
        # gaze_ratio = (gaze_ratio_left_eye+gaze_ratio_right_eye)/2
        # print(frameNo,round(gaze_ratio,2))
        # self.writeData += str(round(frameNo/self.fps,2))+',GR,'+ str(round(gaze_ratio,2))+','+str(int(face.landmarks[30][0]))+','+str(int(face.landmarks[30][1]))+'\n'
        # self.writeData += str(round(frameNo/self.fps,2))+',GR,'+ str(round(gaze_ratio,2))+'\n'

        # if not self.show_landmarks:
        #     return
        # self.visualizer.draw_points(face.landmarks,
        #                             color=(0, 255, 255),
        #                             size=1)

    def _gaze_ratio(self,eyepoints,landmarks,gray):
        # eye_region = np.array([(landmarks.part(eyepoints[0]).x, landmarks.part(eyepoints[0]).y),
        #                         (landmarks.part(eyepoints[1]).x, landmarks.part(eyepoints[1]).y),
        #                         (landmarks.part(eyepoints[2]).x, landmarks.part(eyepoints[2]).y),
        #                         (landmarks.part(eyepoints[3]).x, landmarks.part(eyepoints[3]).y),
        #                         (landmarks.part(eyepoints[4]).x, landmarks.part(eyepoints[4]).y),
        #                         (landmarks.part(eyepoints[5]).x, landmarks.part(eyepoints[5]).y)],dtype=np.int32)
        nose_y = int(landmarks[30][1])
        image_width = self.coordinate[2]-self.coordinate[1]
        
        eye_region = np.array([landmarks[eyepoints[0]],landmarks[eyepoints[1]],landmarks[eyepoints[2]],landmarks[eyepoints[3]],landmarks[eyepoints[4]],landmarks[eyepoints[5]]],dtype=np.int32)
        # height,width = gray.shape
        # mask = np.zeros((height,width), np.uint8)
        # eye = cv2.bitwise_and(gray,gray,mask=mask)
        
        min_x = np.min(eye_region[:,0])
        max_x = np.max(eye_region[:,0])
        min_y = np.min(eye_region[:,1])
        max_y = np.max(eye_region[:,1])
        # print(min_x,max_x,min_y,max_y)
        # gray_eye = eye[min_y:max_y,min_x:max_x]
        # print(gray_eye)
        gray_eye = gray[min_y:max_y,min_x:max_x]
        # print(gray_eye)
        kmeans = KMeans(n_clusters=2).fit(np.array(gray_eye).reshape(-1,1))
        _, threshold_eye = cv2.threshold(gray_eye,int(np.mean(kmeans.cluster_centers_)),255,cv2.THRESH_BINARY)
        height,width = threshold_eye.shape
        # print(height,width)
        # left_side_threshold = threshold_eye[0:height,0:int(width*(1-(nose_y/image_width)))]
        left_side_threshold = threshold_eye[0:height,0:int(width/2)]
        left_side_white = cv2.countNonZero(left_side_threshold)

        # right_side_threshold = threshold_eye[0:height,int(width*(1-(nose_y/image_width))):width]
        right_side_threshold = threshold_eye[0:height,int(width/2):width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        # if right_side_white == 0:
        #     right_side_white = 1
        # gazeRatio = left_side_white/right_side_white
        # # print(gazeRatio)
        # return gazeRatio

        left_side_white_ratio = left_side_white/((max_x-min_x+1)*(max_y-min_y+1))
        right_side_white_ratio = right_side_white/((max_x-min_x+1)*(max_y-min_y+1))
        return left_side_white_ratio,right_side_white_ratio

    def _pointProjects(self,landmarks,dim,frameNo):
        imgCols = dim[0]
        focalLength = imgCols/2
        cameraMatrix = np.array([[focalLength, 0, dim[0]/2],[0, focalLength, dim[1]/2],[0, 0, 1]], dtype=np.float)
        distCoeffs = np.array([0, 0, 0, 0], dtype=np.float)
        rotVector = np.array([0], dtype=np.float)
        translVector = np.array([0], dtype=np.float)
        noseEndPoint3d = np.array([0, 0, 1000], dtype=np.float) ## np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        noseEndPoint2d = np.array([0, 0], dtype=np.float)
        landmarkLoc=landmarks
        twoDImagePoints = np.array([ [landmarkLoc[30]], [landmarkLoc[8]], [landmarkLoc[36]],[landmarkLoc[45]], [landmarkLoc[48]], [landmarkLoc[54]]], dtype=np.float)
        # print(twoDImagePoints)
        modelPoints = self._getModelPoints()
        retval, rotVector, translVector = cv2.solvePnP(modelPoints, twoDImagePoints, cameraMatrix, distCoeffs, rotVector, translVector)
        imgpt = cv2.projectPoints(noseEndPoint3d, rotVector, translVector, cameraMatrix, distCoeffs, noseEndPoint2d)[0]
        # print(imgpt[0][0][0],imgpt[0][0][1])
        self.writeData += str(round(frameNo/self.fps,2))+',PP,'+str(round(imgpt[0][0][0],2))+','+str(round(imgpt[0][0][1],2))+'\n' ## Snigdha Added


    def _getModelPoints(self):
        modelPoints = np.array([[0,0,0],[0, -330, -65],[-225, 170, -135],[225, 170, -135],[-150, -150, -125],[150, -150, -125]], dtype=np.float)
        return modelPoints

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        # self.visualizer.draw_3d_points(face.model3d,
        #                                color=(255, 0, 525),
        #                                size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face,frameNo) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                # self.visualizer.draw_3d_line(
                #     eye.center, eye.center + length * eye.gaze_vector,lw=2)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                # logger.info(
                    # f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
                self.writeData += str(round(frameNo/self.fps,2))+','+key.name.lower()+','+str(round(pitch,2))+','+str(round(yaw,2))+'\n' ## Snigdha Added
                # self.writeData += str(frameNo/self.fps)+','+key.name.lower()+','+','.join(str(x) for x in eye.gaze_vector.tolist())+'\n' ## Snigdha Added
                # self.text_writer.write(str(frameNo/self.fps)+','+key.name.lower()+','+','.join(str(x) for x in eye.gaze_vector.tolist())+'\n') ## Snigdha Added and commented
                # logger.info(f'[{key.name.lower()}] gaze_vector: {eye.gaze_vector}') ## Snigdha commented
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            # self.visualizer.draw_3d_line(
            #     face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            # logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
            # self.text_writer.write(eye.gaze_vector)
            logger.info(f'[face] gaze_vector: {eye.gaze_vector}')
        else:
            raise ValueError
