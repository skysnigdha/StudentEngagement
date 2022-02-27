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


# @profile
def timeSegmentGen(startTime,endTime):
	finalStartTimeList = [startTime[2]]
	finalEndTimeList = [endTime[2]]
	k = 1

	if len(startTime) == 2:
		return finalStartTimeList,finalEndTimeList

	for i in range(3,len(endTime)-1):
		if endTime[i] + 30 >= startTime[i+1]:
			finalEndTimeList[k-1] = endTime[i+1]
		else:
			finalStartTimeList.append(startTime[i+1])
			finalEndTimeList.append(endTime[i+1])
			k += 1
	startTimeList = finalStartTimeList
	endTimeList = finalEndTimeList
	finalStartTimeList = []
	finalEndTimeList = []
	for i in range(len(startTimeList)):
		if endTimeList[i] - startTimeList[i] > 1: #temporal threshold
			finalStartTimeList.append(startTimeList[i])
			finalEndTimeList.append(endTimeList[i])

	# print(finalStartTimeList,finalEndTimeList)
	return finalStartTimeList,finalEndTimeList

def backgroundSubtractor(video_path,coordinate):
	ts1 = datetime.now().timestamp()
	# h = hpy()
	# print(h.heap())
	startTimeList = []
	endTimeList = []
	
	cap = cv2.VideoCapture(video_path)
	x = coordinate[0]
	y = coordinate[1]
	if coordinate[2] <= 0:
		h = int(cap.get(3))+coordinate[2]
	else:
		h = coordinate[2]
	if coordinate[3] <= 0:
		w = int(cap.get(4))+coordinate[3]
	else:
		w = coordinate[3]

	# outvis_name = os.path.basename(video_path).replace('.mp4','_output.avi')
	# outvis_name = os.path.basename(video_path).replace('.avi','_output.avi')
	imwidth = int(cap.get(3)); imheight = int(cap.get(4))
	# outvid = cv2.VideoWriter(outvis_name,cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(5), (imwidth,imheight))
	dataDic = {}
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
	frameNo = 0
	flag = True
	while(1):
	    ret, frame = cap.read()
	    if not ret:
	    	break
	    frame = frame[x:w,y:h]
	    frameNo += 1
	    fgmask = fgbg.apply(frame)
	    # print(fgmask)	
	    dataDic[frameNo] = fgmask
	    img = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
	    median = cv2.medianBlur(img,5)
	    # cv2.imshow('frame',median)

	    whitePixel = np.sum(median == 255)
	    if whitePixel > 1000 and flag: #spacial threshold
	    	startTimeList.append(frameNo)
	    	# print(frameNo,'-')
	    	flag = False
	    elif whitePixel < 1000 and not flag:
	    	endTimeList.append(frameNo)
	    	# print(frameNo-1)
	    	flag = True

	    # outvid.write(median)
	    k = cv2.waitKey(30) & 0xff
	    if k == 27:
	        break

	cap.release()
	cv2.destroyAllWindows()
	# outvid.release()

	ts2 = datetime.now().timestamp()
	# print(ts2-ts1)
	return startTimeList,endTimeList

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
	parser.add_argument('-s', '--std-name-file', dest="studentNameFile", default='studentList.txt', type=str)
	parser.add_argument('-d', '--data-collection-day', dest="day", default='20211205', type=str)
	# parser.add_argument('-c', '--user-coordinate', dest="coordinate", default=[300,600,1024,720], type=int)
	parser.add_argument('-c', '--user-coordinate', dest="coordinateFile", default='lectureCoordinate.txt', type=str)
	parser.add_argument('-f', '--fps', dest="fps", default=30, type=int)
	parser.add_argument('-sl', '--slide-count', dest="slideCount", default=20, type=int)
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
		# if i == 3:
		# 	continue
		for j in range(option.slideCount):
			# if j not in (2,3):
			# 	continue
			videoFileNames = glob.glob(option.videoFile+studentList[i]+'_'+option.day+'/*_'+str(j+1)+'.mp4')
			# print(coordinateList)
			
			for videoFileName in videoFileNames:
				if not os.path.isfile(videoFileName):
					continue
				# print(videoFileName)
				startTimeList,endTimeList = backgroundSubtractor(videoFileName,	coordinateList[studentList[i]])
				# print(startTimeList,endTimeList)
				if len(endTimeList) > 2:
					startTimeList,endTimeList = timeSegmentGen(startTimeList,endTimeList)
					print(videoFileName.split('/')[-1].split('.')[0].split('-')[-1],startTimeList,endTimeList)
	return

if __name__ == '__main__':
	main()
