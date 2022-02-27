import numpy as np
import cv2
from datetime import datetime
import argparse
import os,sys
# from skimage.measure import structural_similarity as ssim
# from skimage.metrics import structural_similarity as ssim
# from skimage import measure
import glob
# from guppy import hpy
# import psutil
# from memory_profiler import profile
# np.set_printoptions(threshold=sys.maxsize)

def lookAtScreen(filename,model,coordinate):
	outData = ''
	cap = cv2.VideoCapture(filename)
	x = coordinate[0]
	y = coordinate[1]
	h = coordinate[2]
	w = coordinate[3]

	face_cascade = cv2.CascadeClassifier(model)
	frameNo = 0
	while(1):
		ret, frame = cap.read()
		if not ret:
			break
		frame = frame[x:w,y:h]
		frameNo += 1
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray,1.1,4)
		if len(faces)==1:
			# print(frameNo,1)
			outData += str(frameNo) + ',' + str(1) + '\n'
		else:
			# print(frameNo,0)
			outData += str(frameNo) + ',' + str(0) + '\n'
	return outData


def readCoordinateFile(filename):
	inFile = open(filename,'r')
	coordinateData = {}
	for line in inFile:
		words = line.strip('\r\n').split(',')
		# print(words)
		coordinateData[words[0]] = [int(words[1]),int(words[2]),int(words[3]),int(words[4])]
	return coordinateData

def readFile(filename):
	inFile = open(filename,'r')
	nameList = []
	for line in inFile:
		nameList.append(line.strip('\r\n'))
	return nameList

def parseArg(): 
	parser = argparse.ArgumentParser(description = "Student Engagement")
	parser.add_argument('-v', '--video-dir', dest="videoFile", default='videoData/', type=str)
	parser.add_argument('-o', '--match-output-dir', dest="outFile", default='LookingAtScreen/', type=str)
	parser.add_argument('-s', '--std-name-file', dest="studentNameFile", default='studentList.txt', type=str)
	parser.add_argument('-d', '--data-collection-day', dest="day", default='20211205', type=str)
	# parser.add_argument('-c', '--user-coordinate', dest="coordinate", default=[300,600,1024,720], type=int)
	parser.add_argument('-c', '--user-coordinate', dest="coordinateFile", default='viewerCoordinate.txt', type=str)
	parser.add_argument('-f', '--fps', dest="fps", default=30, type=int)
	parser.add_argument('-sl', '--slide-count', dest="slideCount", default=4, type=int)
	parser.add_argument('-m', '--facial-model', dest="facialModel", default="haarcascade_frontalface_default.xml", type=str)
	return parser.parse_args()

def main():
	option = parseArg()
	studentList = readFile(option.studentNameFile)
	coordinateList = readCoordinateFile(option.day + '_' +option.coordinateFile)
	# print(studentList)

	for i in range(len(studentList)):
		if not os.path.isdir(option.videoFile+studentList[i]+'_'+option.day):
			continue
		# if i != 3:
		# 	continue
		# if i not in (3,35,36):
		# 	continue
		for j in range(option.slideCount):
			# if j not in (2,3):
			# 	continue
			videoFileNames = glob.glob(option.videoFile+studentList[i]+'_'+option.day+'/*_'+str(j+1)+'.mp4')
			# print(coordinateList)
			print(studentList[i]+'_'+str(j+1))
			for videoFileName in videoFileNames:
				if not os.path.isfile(videoFileName):
					continue
				userCoordinate = [coordinateList[studentList[i]][0],coordinateList[studentList[i]][1],coordinateList[studentList[i]][2],coordinateList[studentList[i]][3]]
				outfilename = option.outFile + videoFileName.split('/')[-1].split('.')[0]+'.txt' 
				outData = lookAtScreen(videoFileName,option.facialModel,userCoordinate)
				outputFile = open(outfilename,'w')
				outputFile.write(outData)
				outputFile.close()
	return

if __name__ == '__main__':
	main()
