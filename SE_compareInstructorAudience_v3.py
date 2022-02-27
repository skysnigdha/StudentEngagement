import argparse,os,glob,cv2
import numpy as np
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y



def readOnScreenFile(filename):
	inFile = open(filename,'r')
	screenData = {}
	for line in inFile:
		words = line.strip('\r\n').split(',')
		# print(words)
		screenData[int(words[0])] =int(words[1])
	inFile.close()
	return screenData

def screenProcessing(filename):
	t = []

	screenData = readOnScreenFile(filename)
	for screenKey in sorted(screenData.keys()):
		if screenData[screenKey] == 1: 
			t.append(screenKey)
	return t 


# def readPPFile(filename):
# 	inFile = open(filename,'r')
# 	ppData = {}
# 	for line in inFile:
# 		words = line.strip('\r\n').split(',')
# 		# if len(words) == 3:
# 		if words[1] == 'PP': 
# 			if 'pp' not in ppData.keys():
# 				ppData['pp'] = {}
# 				ppData['pp'][0] = []
# 				ppData['pp'][1] = []
# 				ppData['pp'][2] = []
# 			ppData['pp'][0].append(float(words[2]))
# 			ppData['pp'][1].append(float(words[3]))
# 			ppData['pp'][2].append(float(words[0]))
# 	inFile.close()
# 	return ppData

# def projectPointProcessing(filename,thrshldx,thrshldy,cutoff,fs,order):
# 	y = []
# 	t = []

# 	ppData = readPPFile(filename)
# 	for ppKey in sorted(ppData.keys()):
# 		if ppKey == 'pp':
# 			for k in range(len(ppData[ppKey][0])):
# 				if 0.<=ppData[ppKey][0][k]<=thrshldx and 0.<=ppData[ppKey][1][k]<=thrshldy: 
# 					y.append(ppData[ppKey][1][k]) 
# 					t.append(ppData[ppKey][2][k])
# 	y = np.array(y)
# 	y_filter = butter_lowpass_filter(y, cutoff, fs, order)

# 	# print(y_filter,t)
# 	return(y_filter,t)

def compareLecture(video_path,gvideo_path,gstartTime,startTime,similarityThreshold,fs,delay):
	compareResult = []
	compareFrame = []
	gcap = cv2.VideoCapture(gvideo_path)
	
	# print(gcap.get(0),gcap.get(1),gcap.get(2),gcap.get(5),gcap.get(6))

	gimwidth = int(gcap.get(3)); gimheight = int(gcap.get(4))
	# print(gimwidth,gimheight)

	
	
	for i in range(len(gstartTime)):
		startFrame = -1
		j = 0
		probableFrameSet = []
		probableSimilaritySet = []
		while(j<len(startTime)):
			if gstartTime[i]-fs*delay <= startTime[j] <= gstartTime[i]+fs*delay:
				startFrame = startTime[j]
				probableFrameSet.append(startTime[j])
				# print(startFrame)
				

				gcap.set(1,gstartTime[i])
				while True:	
					gret, gframe = gcap.read()
					if not gret:
						break
					# gGrayframe = cv2.cvtColor(gframe,cv2.COLOR_BGR2GRAY)
					ghist = cv2.calcHist([gframe],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
					ghist = cv2.normalize(ghist,ghist).flatten()
					break

				cap = cv2.VideoCapture(video_path)
				if startFrame == -1:
					startFrame = gstartTime[i]
				cap.set(1,startFrame)

				imwidth = int(cap.get(3)); imheight = int(cap.get(4))
				# print(imwidth,imheight)
				count = 0
				
				minDist = 9999

				while(1):
					ret, frame = cap.read()
					count += 1
					if not ret:
						break

					# grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
					hist = cv2.calcHist([frame],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
					hist = cv2.normalize(hist,hist).flatten()

					dist = cv2.compareHist(ghist,hist,cv2.HISTCMP_CHISQR)
					# print(dist)
					if minDist > dist:
						minDist = dist
					if count == fs*delay:
						break
				# print(minDist)
				probableSimilaritySet.append(minDist)
				# if minDist <= similarityThreshold:
				# 	probableSimilaritySet.append(1)
				# else:
				# 	probableSimilaritySet.append(0)

				
			j += 1
		
		if len(probableFrameSet) == 0:
			compareFrame.append(-1)
			gcap.set(1,gstartTime[i])
			while True:	
				gret, gframe = gcap.read()
				if not gret:
					break
				# gGrayframe = cv2.cvtColor(gframe,cv2.COLOR_BGR2GRAY)
				ghist = cv2.calcHist([gframe],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
				ghist = cv2.normalize(ghist,ghist).flatten()
				break

			cap = cv2.VideoCapture(video_path)
			if startFrame == -1:
				startFrame = gstartTime[i]
			cap.set(1,startFrame)

			imwidth = int(cap.get(3)); imheight = int(cap.get(4))
			# print(imwidth,imheight)
			count = 0
			
			minDist = 9999

			while(1):
				ret, frame = cap.read()
				count += 1
				if not ret:
					break

				# grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
				hist = cv2.calcHist([frame],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
				hist = cv2.normalize(hist,hist).flatten()

				dist = cv2.compareHist(ghist,hist,cv2.HISTCMP_CHISQR)
				# print(dist)
				if minDist > dist:
					minDist = dist
				if count == fs*delay:
					break
			if minDist <= similarityThreshold:
				compareResult.append(1)
			else:
				compareResult.append(0)

		else:
			probableSimilaritySet = np.array(probableSimilaritySet)
			# print(probableSimilaritySet,probableFrameSet)
			k = np.argmin(probableSimilaritySet)
			if probableSimilaritySet[k] <= similarityThreshold:
				compareResult.append(1)
			else:
				compareResult.append(0)
			compareFrame.append(probableFrameSet[k])

	return compareResult,compareFrame

def readForegroundData(foregroundFile):
	if not os.path.isfile(foregroundFile):
		# print('No foreground data...')
		return
	data = {}
	inFile = open(foregroundFile,'r')
	for line in inFile:
		words = line.strip('\r\n').split(' [')
		user = words[0].replace(' ','')
		# if user.split('_')[0] == instructor:
		startList = [int(v) for v in words[1].replace(']','').split(',')]
		endList = [int(v) for v in words[2].replace(']','').split(',')]
		data[user] = [startList,endList]
	inFile.close()
	return data


def readFile(filename):
	inFile = open(filename,'r')
	nameList = []
	for line in inFile:
		nameList.append(line.strip('\r\n'))
	return nameList

def parseArg():
	parser = argparse.ArgumentParser(description = "Student Engagement")
	parser.add_argument('-v', '--video-datapath', dest="videoDataPath", default='videoData/', type=str)
	parser.add_argument('-fd', '--face-datapath', dest="faceDataPath", default='LookingAtScreen/', type=str)#'outputData/'
	parser.add_argument('-s', '--std-name-file', dest="studentNameFile", default='studentList.txt', type=str)
	parser.add_argument('-d', '--data-collection-day', dest="day", default='2021-12-05', type=str)
	# # parser.add_argument('-c', '--user-coordinate', dest="coordinate", default=[0,0,950,750], type=int)#[0,0,600,330]
	# parser.add_argument('-c', '--user-coordinate', dest="coordinateFile", default='viewerCoordinate.txt', type=str)
	parser.add_argument('-fs', '--fps', dest="fs", default=30, type=int)
	parser.add_argument('-sc', '--slide-count', dest="slideCount", default=4, type=int)
	# parser.add_argument('-sp', '--scale-percentage', dest="scalePercentage", default=100, type=int)
	parser.add_argument('-i', '--instructor', dest="instructor", default='SC', type=str)
	parser.add_argument('-ff', '--forground-file', dest="foregroundFile", default='_ForegroundExtractor.txt', type=str)
	parser.add_argument('-fbs', '--forground-background-similarity-threshold', dest="foreBackgroundThreshold", default=20, type=int)
	parser.add_argument('-fpx', '--face-position-threshold-x', dest="facePositionThresholdX", default=300, type=int)
	parser.add_argument('-fpy', '--face-position-threshold-y', dest="facePositionThresholdY", default=300, type=int)
	parser.add_argument('-de', '--tolerable-delay', dest="delay", default=4, type=int)
	parser.add_argument('-o', '--low-pass-order', dest="order", default=6, type=int)
	parser.add_argument('-c', '--cut-off', dest="cutoff", default=3.667, type=int)
	return parser.parse_args()

def main():
	option = parseArg()
	studentList = readFile(option.studentNameFile)
	foregroundFile = option.day.replace('-','') + option.foregroundFile
	foregroundData = readForegroundData(foregroundFile) 

	# print(foregroundData)
	
	for j in range(option.slideCount):
		# if j not in (2,3):
		# 	continue
		for i in range(len(studentList)):
			# if i != 3:
			# 	continue
			if studentList[i] == option.instructor:
				startTime = foregroundData[studentList[i]+'_'+str(j+1)][0]
				endTime = foregroundData[studentList[i]+'_'+str(j+1)][1]
				for k in range(len(endTime)):
					endTime[k] += int(option.fs*option.delay/2)
				print(studentList[i]+'_'+str(j+1),startTime,endTime)
				continue 
			# if i!=1 or j!=3:
			# 	continue
			videoPath = glob.glob(option.videoDataPath+studentList[i]+'_'+option.day.replace('-','')+'/*-'+studentList[i]+'_'+str(j+1)+'.mp4')
			gvideoPath = glob.glob(option.videoDataPath+option.instructor+'_'+option.day.replace('-','')+'/*-'+option.instructor+'_'+str(j+1)+'.mp4')

			if len(videoPath) == 0 or len(gvideoPath) == 0:
				# print(studentList[i]+'_'+str(j+1), 'No')
				continue
			# print(studentList[i]+'_'+str(j+1))
			if studentList[i]+'_'+str(j+1) not in foregroundData.keys():
				continue
			startTime = foregroundData[studentList[i]+'_'+str(j+1)][0]
			gstartTime = foregroundData[option.instructor+'_'+str(j+1)][0]
			endTime = foregroundData[studentList[i]+'_'+str(j+1)][1]
			gendTime = foregroundData[option.instructor+'_'+str(j+1)][1]
			# print(gstartTime,startTime)

			foreBackCompare,foreBackFrame = compareLecture(videoPath[0], gvideoPath[0], gstartTime, startTime,option.foreBackgroundThreshold,option.fs,option.delay)
			# print(foreBackCompare,foreBackFrame)

			if np.all(np.array(foreBackCompare)==0):
				# print(studentList[i]+'_'+str(j+1), 'No-Fore')
				continue

			foreBackFrameStart = []
			foreBackFrameEnd = []
			for k in range(len(startTime)):
				for l in foreBackFrame:
					if startTime[k] == l:
						foreBackFrameStart.append(startTime[k])
						foreBackFrameEnd.append(endTime[k])
			# print(foreBackFrameStart,foreBackFrameEnd)

			faceDataPath = glob.glob(option.faceDataPath+option.day+'*-'+studentList[i]+'_'+str(j+1)+'.txt')
			gfaceDataPath = glob.glob(option.faceDataPath+option.day+'*-'+option.instructor+'_'+str(j+1)+'.txt')

			if len(faceDataPath) == 0 or len(gfaceDataPath) == 0:
				# print(studentList[i]+'_'+str(j+1), 'No-Face')
				continue

			gt = screenProcessing(gfaceDataPath[0])
			t = screenProcessing(faceDataPath[0])

			# flag = False
			# if len(foreBackFrameStart) > 0:
			# 	for l in t:
			# 		# print(l)
			# 		for k in range(len(foreBackFrameStart)):
			# 			if foreBackFrameStart[k]<=l<=foreBackFrameEnd[k]:
			# 				# print(l)
			# 				flag = True
			# print(flag)

			selectedStartList = []
			selectedEndList = []
			if len(foreBackFrameStart) > 0:
				for k in range(len(foreBackFrameStart)):
					flag = False
					for l in range(len(t)):
						if foreBackFrameStart[k] <= t[l] <= foreBackFrameEnd[k]+option.fs*option.delay/2:
							if not flag or t[l] - selectedEndList[len(selectedEndList)-1] > option.fs*option.delay/2:
								flag = True
								selectedStartList.append(t[l])
								selectedEndList.append(t[l])
							else:
								selectedEndList[len(selectedEndList)-1] =t[l]
			if len(selectedEndList) > 0:
				print(studentList[i]+'_'+str(j+1),selectedStartList,selectedEndList)
			# else:
			# 	print("Not looking at screen")



if __name__ == '__main__':
	main()