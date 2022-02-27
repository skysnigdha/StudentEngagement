import argparse,os,glob,cv2
import numpy as np
import scipy as sp
from scipy.signal import butter, lfilter, freqz
from scipy.fftpack import fft,fftfreq
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

def butter_lowpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y

def plotData(user,start,end,dev,cutoff,fs,order):
	# dev_filter = butter_lowpass_filter(dev, cutoff, fs)
	# dev_filter_freq = fft(dev_filter)
	dev_filter_freq = fft(dev)

	N = len(dev)
	T = 1/fs
	xf = fftfreq(N, T)[:N//2]
	plt.plot(xf, 2.0/N * np.abs(dev_filter_freq[:N//2]))
	plt.xlabel('frequency')
	plt.grid()
	plt.legend()

	plt.savefig('plot/'+user+'_'+str(start)+'_'+str(end)+'.png')
	plt.close()


def deviationRangeCompute(user,gdevData,dev,cutoff,fs,order):
	devData = {}
	lastKey = -1
	for key in sorted(gdevData.keys()):
		devList =[]
		startFlag = False
		start = -1
		for k in range(key,gdevData[key][1]+1):
			if k in dev.keys():
				if not startFlag:
					start = k
					startFlag = True 
				devList += dev[k]
		while(k in dev.keys()):
			devList += dev[k]
			k+=1
		if len(devList) > 0:
			devData[key] = [devList,gdevData[key][1]]
			# plotData(user, key, gdevData[key][1], devList, cutoff, fs, order)
		lastKey = key
	if len(devList) > 0:
		devData[lastKey] = [devList,gdevData[lastKey][1]]
	return devData

def groundDeviationRangeCompute(user,dev,cutoff,fs,order):
	devData = {}
	prevKey = -1
	start = -1
	end = -1
	devList = []
	for key in sorted(dev.keys()):

		if start == -1:
			prevKey = key
			devList += dev[key]
			start = key
			end = key
			# print(devList)
		elif prevKey+1 == key:
			prevKey = key
			devList += dev[key]
			end = key
		else:
			# print(devList)
			devData[start] = [devList,end]
			# plotData(user, start, end, devList, cutoff, fs, order)
			prevKey = key
			devList = dev[key]
			start = key
			end = key
	if len(devList) > 0:
		devData[start] = [devList,end]	
	# print(devList)
	return devData


def pointDeviationCompute(time,pp):
	dev = {}
	ppList = []
	for i in range(len(time)):
		if int(time[i]/30) not in dev.keys() and len(ppList) > 0:
			dev[int(time[i-1]/30)] = ppList
			ppList = []
		ppList.append(pp[i])
	return dev

def readFaceMatchData(faceFile):
	if not os.path.isfile(faceFile):
		# print('No foreground data...')
		return
	data = {}
	inFile = open(faceFile,'r')
	for line in inFile:
		words = line.strip('\r\n').split(' [')
		user = words[0].replace(' ','')
		# if user.split('_')[0] == instructor:
		startList = [int(v) for v in words[1].replace(']','').split(',')]
		endList = [int(v) for v in words[2].replace(']','').split(',')]
		data[user] = [startList,endList]
	inFile.close()
	return data

def readPPFile(filename):
	inFile = open(filename,'r')
	timeData = []
	ppData = []
	lGazeData = []
	rGazeData = []
	for line in inFile:
		words = line.strip('\r\n').split(',')
		if words[1] == 'PP': 
			timeData.append(int(float(words[0])*30))
			ppData.append(float(words[2]))
		if words[1] == 'leye':
			lGazeData.append(float(words[3]))
		elif words[1] == 'reye':
			rGazeData.append(float(words[3]))
	inFile.close()
	return timeData,ppData,lGazeData,rGazeData


def readFile(filename):
	inFile = open(filename,'r')
	nameList = []
	for line in inFile:
		nameList.append(line.strip('\r\n'))
	return nameList

def parseArg():
	parser = argparse.ArgumentParser(description = "Student Engagement")
	# parser.add_argument('-v', '--video-datapath', dest="videoDataPath", default='videoData/', type=str)
	parser.add_argument('-fd', '--facematch-datapath', dest="faceDataPath", default='faceMatchData1/', type=str)#'outputData/'
	parser.add_argument('-s', '--std-name-file', dest="studentNameFile", default='studentList.txt', type=str)
	parser.add_argument('-d', '--data-collection-day', dest="day", default='2021-12-05', type=str)
	# # parser.add_argument('-c', '--user-coordinate', dest="coordinate", default=[0,0,950,750], type=int)#[0,0,600,330]
	# parser.add_argument('-c', '--user-coordinate', dest="coordinateFile", default='viewerCoordinate.txt', type=str)
	parser.add_argument('-fs', '--fps', dest="fs", default=30, type=int)
	parser.add_argument('-sc', '--slide-count', dest="slideCount", default=4, type=int)
	# parser.add_argument('-sp', '--scale-percentage', dest="scalePercentage", default=100, type=int)
	parser.add_argument('-i', '--instructor', dest="instructor", default='SC', type=str)
	parser.add_argument('-ff', '--face-file', dest="faceFile", default='_InstructorAudienceFacialMatch_v3.txt', type=str)
	# parser.add_argument('-ff', '--forground-file', dest="foregroundFile", default='_ForegroundExtractor.txt', type=str)
	# parser.add_argument('-fbs', '--forground-background-similarity-threshold', dest="foreBackgroundThreshold", default=20, type=int)
	# parser.add_argument('-fpx', '--face-position-threshold-x', dest="facePositionThresholdX", default=300, type=int)
	# parser.add_argument('-fpy', '--face-position-threshold-y', dest="facePositionThresholdY", default=300, type=int)
	parser.add_argument('-de', '--tolerable-delay', dest="delay", default=4, type=int)
	parser.add_argument('-or', '--low-pass-order', dest="order", default=6, type=int)
	parser.add_argument('-c', '--cut-off', dest="cutoff", default=3.667, type=int)
	return parser.parse_args()

def main():
	option = parseArg()
	studentList = readFile(option.studentNameFile)
	# foregroundFile = option.day.replace('-','') + option.foregroundFile
	# foregroundData = readForegroundData(foregroundFile) 
	faceFile = option.day.replace('-','') + option.faceFile
	faceData = readFaceMatchData(faceFile)
	# print(foregroundData)
	for j in range(option.slideCount):
		for i in range(len(studentList)):
			if studentList[i] == option.instructor:
				continue 
			# if i!=1 or j!=3:
			# 	continue

			if studentList[i]+'_'+str(j+1) in faceData:
				startTime = faceData[studentList[i]+'_'+str(j+1)][0]
				endTime = faceData[studentList[i]+'_'+str(j+1)][1]
			else:
				continue

			gstartTime = faceData[option.instructor+'_'+str(j+1)][0]
			gendTime = faceData[option.instructor+'_'+str(j+1)][1]			

			faceMatchPath = glob.glob(option.faceDataPath+option.day+'*-'+studentList[i]+'_'+str(j+1)+'.txt')
			gfaceMatchPath = glob.glob(option.faceDataPath+option.day+'*-'+option.instructor+'_'+str(j+1)+'.txt')

			if len(faceMatchPath) == 0 or len(gfaceMatchPath) == 0:
				# print(studentList[i]+'_'+str(j+1), 'No')
				continue
			# print(studentList[i]+'_'+str(j+1))
			timeData,ppData,lGazeData,rGazeData = readPPFile(faceMatchPath[0])
			gtimeData,gppData,glGazeData,grGazeData = readPPFile(gfaceMatchPath[0])

			# print(ppData,gppData)
			# print(startTime,gstartTime)

			gcount = 0
			for k in range(len(gstartTime)):
				for l in gtimeData:			
					if gstartTime[k] <= l <= gendTime[k]:
						gcount +=1
						break
			count = 0
			for k in range(len(startTime)):
				for l in timeData:			
					if startTime[k] <= l <= endTime[k]:
						count +=1
						break
			# print(round(gcount/len(gstartTime),2),round(count/len(gstartTime),2))

			if count/len(gstartTime) >= 0.5:
				dev = pointDeviationCompute(timeData,ppData)
				gdev = pointDeviationCompute(gtimeData,gppData)

				# print(dev,gdev)
				gdevData = groundDeviationRangeCompute(option.instructor+'_'+str(j+1),gdev,option.cutoff,option.fs,option.order)
				# print(gdevData)
				devData = deviationRangeCompute(studentList[i]+'_'+str(j+1),gdevData,dev,option.cutoff,option.fs,option.order)
				# print(devData)

				if not devData:
					continue

				energy = []
				genergy = []
				# for key in sorted(devData.keys()):
				# 	# print(key,np.array(gdevData[key][0]))
				# 	e = np.sum(np.array(devData[key][0])**2)
				# 	ge = np.sum(np.array(gdevData[key][0])**2)
					
				# 	energy.append(e)
				# 	genergy.append(ge)
				# 	# print(e,ge)
				# 	# print(key,round(e,2),round(ge,2))
				for key in sorted(dev.keys()):
					# print(key,np.array(gdevData[key][0]))
					# e = np.sum(np.array(devData[key][0])**2)
					# ge = np.sum(np.array(gdevData[key][0])**2)
					# print(key)
					e = np.sum(np.array(dev[key])**2)
					ge = np.sum(np.array(gdev[key])**2)
					
					energy.append(e)
					genergy.append(ge)
					# print(e,ge)
					# print(key,round(e,2),round(ge,2))
				stat,p =ttest_ind(energy,genergy)
				# print(stat,p)
				if p > 0.001:
					# print('same')
					print(studentList[i]+'_'+str(j+1),'Y')
				else:
					# print('different')
					print(studentList[i]+'_'+str(j+1),'N')
			else:
				print(studentList[i]+'_'+str(j+1),'N')

		# 	gcount = 0
		# 	count = 0
		# 	for key in sorted(gdevData.keys()):
		# 		gcount += 1
		# 		if key in devData.keys():
		# 			count += 1
		# 			# print(key,round(gdevData[key][0],2),round(devData[key][0],2))
		# 	ground = [k[0] for k in list(gdevData.values())]
		# 	current = [k[0] for k in list(devData.values())]
		# 	print(round(np.std(ground),2),round(np.std(current),2))
		# 	# print(round(count/gcount,2))

		# # 	break
		# # break
				



if __name__ == '__main__':
	main()
