import argparse
import glob
import logging
import datetime

from ptgaze import get_default_config
# from ptgaze.demo import Demo
import demo
from ptgaze.utils import update_default_config, update_config
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as moviedi
import os

# def extensionConvertMP4(filename):
# 	clip = moviedi.VideoFileClip(filename)
# 	outFilename = filename.split('.')[0]+'.mp4'
# 	clip.write_videofile(outFilename)
# 	return outFilename


# def cuttingVideo(filename,timeList):
# 	if filename.split('.')[1] not in ('mp4'):
# 		filename = extensionConvertMP4(filename)
# 		print(filename)
# 	outFileList = []
# 	for i in range(len(timeList)):
# 		outFilename = filename.split('.')[0]+'_'+timeList[i][3]+'.'+filename.split('.')[1]
# 		# outFilename = filename.split('.')[0]+'_i.'+filename.split('.')[1]
# 		# outFilename = filename.split('.')[0]+'_'+timeList[i][0]+'.'+filename.split('.')[1]
# 		ffmpeg_extract_subclip(filename, timeList[i][1], timeList[i][2], targetname=outFilename)
# 		outFileList.append(outFilename)
# 	return outFileList

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
# def readTimeFile(filename): ## modified in v4
# 	inFile = open(filename,'r')
# 	startTimeList = []
# 	endTimeList = []
# 	segmentList = []
# 	slideList = []
# 	for line in inFile:
# 		words = line.strip('\r\n').split(' ')
# 		segmentList.append(words[0])
# 		startTime =(int(words[0].split(':')[0])*60+int(words[0].split(':')[1]))*60+int(words[0].split(':')[2])
# 		startTimeList.append(startTime)
# 		endTime = (int(words[1].split(':')[0])*60+int(words[1].split(':')[1]))*60+int(words[1].split(':')[2])
# 		endTimeList.append(endTime)
# 		slideList.append(words[2])
# 	timeList = list(zip(segmentList,startTimeList,endTimeList,slideList))
# 	return timeList

def parseArg(): 
	parser = argparse.ArgumentParser(description = "Student Engagement")
	parser.add_argument(
		'--config',
		type=str,
		help='Config file for YACS. When using a config file, all the other '
		'commandline arguments are ignored. '
		'See https://github.com/hysts/pytorch_mpiigaze_demo/configs/demo_mpiigaze.yaml'
	)
	parser.add_argument(
		'--mode',
		type=str,
		default='eye',
		choices=['eye', 'face'],
		help='With \'eye\', MPIIGaze model will be used. With \'face\', '
		'MPIIFaceGaze model will be used. (default: \'eye\')')
	parser.add_argument(
		'--face-detector',
		type=str,
		default='dlib',
		choices=['dlib', 'face_alignment_dlib', 'face_alignment_sfd'],
		help='The method used to detect faces and find face landmarks '
		'(default: \'dlib\')')
	parser.add_argument('--device',
						type=str,
						choices=['cpu', 'cuda'],
						default = 'cpu',
						help='Device used for model inference.')
	parser.add_argument('--image',
						type=str,
						help='Path to an input image file.')
	parser.add_argument('--video',
						type=str,
						help='Path to an input video file.')
	parser.add_argument(
		'--camera',
		type=str,
		help='Camera calibration file. '
		'See https://github.com/hysts/pytorch_mpiigaze_demo/ptgaze/data/calib/sample_params.yaml'
	)
	parser.add_argument(
		'--output-dir',
		'-o',
		default='outputData',
		type=str,
		help='If specified, the overlaid video will be saved to this directory.'
	)
	parser.add_argument('--ext',
						'-e',
						type=str,
						choices=['avi', 'mp4'],
						help='Output video file extension.')
	parser.add_argument(
		'--no-screen',
		action='store_true',
		help='If specified, the video is not displayed on screen, and saved '
		'to the output directory.')
	parser.add_argument('--debug', action='store_true')

	parser.add_argument('-v', '--video-dir', dest="videoFile", default='videoData/', type=str)
	parser.add_argument('-s', '--std-name-file', dest="studentNameFile", default='studentList.txt', type=str)
	parser.add_argument('-d', '--data-collection-day', dest="day", default='2021-12-05', type=str)
	# parser.add_argument('-c', '--user-coordinate', dest="coordinate", default=[0,0,950,750], type=int)#[0,0,600,330]
	parser.add_argument('-c', '--user-coordinate', dest="coordinateFile", default='viewerCoordinate.txt', type=str)
	parser.add_argument('-ff', '--face-file', dest="faceFile", default='_InstructorAudienceFacialMatch_v3.txt', type=str)
	parser.add_argument('-f', '--fps', dest="fps", default=30, type=int)
	parser.add_argument('-sc', '--slide-count', dest="slideCount", default=5, type=int)
	parser.add_argument('-sp', '--scale-percentage', dest="scalePercentage", default=100, type=int)
	return parser.parse_args()



def main():
	option = parseArg()
	coordinateList = readCoordinateFile(option.day.replace('-','') + '_' +option.coordinateFile)
	studentList = readFile(option.studentNameFile)
	print(studentList)

	faceFile = option.day.replace('-','') + option.faceFile
	faceData = readFaceMatchData(faceFile) 
	for j in range(option.slideCount):
		# if j not in (2,3):
		# 	continue
		for i in range(len(studentList)):
			if not os.path.isdir(option.videoFile+studentList[i]+'_'+option.day.replace('-','')):
				continue
			# if i != 3:
			# 	continue
			# videoFileNames = glob.glob(option.videoFile+studentList[i]+'_'+option.day+'/*_*.mp4')
			videoFileNames = glob.glob(option.videoFile+studentList[i]+'_'+option.day.replace('-','')+'/*-'+studentList[i]+'_'+str(j+1)+'.mp4')

			print(videoFileNames)
			userCoordinate = [coordinateList[studentList[i]][0],coordinateList[studentList[i]][1],coordinateList[studentList[i]][2],coordinateList[studentList[i]][3]]
			
			if studentList[i]+'_'+str(j+1) in faceData:
				startTime = faceData[studentList[i]+'_'+str(j+1)][0]
				endTime = faceData[studentList[i]+'_'+str(j+1)][1]
			else:
				continue

			if i == 2:
				scalePercentage = 100
			else:
				scalePercentage = option.scalePercentage
			
			processStartTime = datetime.datetime.now()
			option.video = videoFileNames[0]
			config = get_default_config()
			if option.config:
				config.merge_from_file(option.config)
				if (option.device or option.image or option.video or option.camera
						or option.output_dir or option.ext or option.no_screen):
					raise RuntimeError(
						'When using a config file, all the other commandline '
						'arguments are ignored.')
				if config.demo.image_path and config.demo.video_path:
					raise ValueError(
						'Only one of config.demo.image_path or config.demo.video_path '
						'can be specified.')
			else:
				update_default_config(config, option)

			update_config(config)
			# logger.info(config)

			demo1 = demo.Demo(config,[],userCoordinate,scalePercentage,option.fps,startTime,endTime)
			demo1.run()
			processEndTime =datetime.datetime.now()
			print(processEndTime-processStartTime)
				


if __name__ == '__main__':
	main()
