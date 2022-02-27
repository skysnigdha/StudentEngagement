import argparse,glob

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
	count = 0

	screenData = readOnScreenFile(filename)
	for screenKey in sorted(screenData.keys()):
		if screenData[screenKey] == 1: 
			count += 1
	return count

def readFile(filename):
	inFile = open(filename,'r')
	nameList = []
	for line in inFile:
		nameList.append(line.strip('\r\n'))
	return nameList

def parseArg():
	parser = argparse.ArgumentParser(description = "Student Engagement")
	# parser.add_argument('-v', '--video-datapath', dest="videoDataPath", default='videoData/', type=str)
	parser.add_argument('-fd', '--face-datapath', dest="faceDataPath", default='LookingAtScreen/', type=str)#'outputData/'
	parser.add_argument('-s', '--std-name-file', dest="studentNameFile", default='studentList.txt', type=str)
	parser.add_argument('-d', '--data-collection-day', dest="day", default='2021-12-05', type=str)
	# # parser.add_argument('-c', '--user-coordinate', dest="coordinate", default=[0,0,950,750], type=int)#[0,0,600,330]
	# parser.add_argument('-c', '--user-coordinate', dest="coordinateFile", default='viewerCoordinate.txt', type=str)
	parser.add_argument('-fs', '--fps', dest="fs", default=30, type=int)
	parser.add_argument('-sc', '--slide-count', dest="slideCount", default=4, type=int)
	# parser.add_argument('-sp', '--scale-percentage', dest="scalePercentage", default=100, type=int)
	parser.add_argument('-i', '--instructor', dest="instructor", default='SC', type=str)
	# parser.add_argument('-ff', '--forground-file', dest="foregroundFile", default='_ForegroundExtractor.txt', type=str)
	# parser.add_argument('-fbs', '--forground-background-similarity-threshold', dest="foreBackgroundThreshold", default=20, type=int)
	# parser.add_argument('-fpx', '--face-position-threshold-x', dest="facePositionThresholdX", default=300, type=int)
	# parser.add_argument('-fpy', '--face-position-threshold-y', dest="facePositionThresholdY", default=300, type=int)
	# parser.add_argument('-de', '--tolerable-delay', dest="delay", default=4, type=int)
	# parser.add_argument('-o', '--low-pass-order', dest="order", default=6, type=int)
	# parser.add_argument('-c', '--cut-off', dest="cutoff", default=3.667, type=int)
	parser.add_argument('-tf', '--total-frame', dest="totalFrame", default=29100, type=int)
	return parser.parse_args()

def main():
	option = parseArg()
	studentList = readFile(option.studentNameFile)

	for j in range(option.slideCount):
		# if j not in (2,3):
		# 	continue
		for i in range(len(studentList)):
			# if i != 3:
			# 	continue
			if studentList[i] == option.instructor:
				continue
			faceDataPath = glob.glob(option.faceDataPath+option.day+'*-'+studentList[i]+'_'+str(j+1)+'.txt')
			gfaceDataPath = glob.glob(option.faceDataPath+option.day+'*-'+option.instructor+'_'+str(j+1)+'.txt')

			
			gt = screenProcessing(gfaceDataPath[0])
			if len(faceDataPath) == 0:
				# print(studentList[i]+'_'+str(j+1), round(gt/option.totalFrame,2), 0.)
				continue
			t = screenProcessing(faceDataPath[0])
			# print(studentList[i]+'_'+str(j+1), round(gt/option.totalFrame,2), round(t/option.totalFrame,2))
			if (t/gt) >= 0.5:
				print(studentList[i]+'_'+str(j+1), "Y")
			else:
				print(studentList[i]+'_'+str(j+1), "N")



if __name__ == '__main__':
	main()