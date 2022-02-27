import argparse,os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def cuttingVideo(filename,startTime,duration):
	outFilename = filename.split('.')[0]+'_1.'+filename.split('.')[1]
	ffmpeg_extract_subclip(filename, startTime, duration, targetname=outFilename)
	return

def parseArg():
	parser = argparse.ArgumentParser(description = "cutting video")
	parser.add_argument('-v', '--video-file', dest="videoFile", default='videoData/test.mp4', type=str)
	parser.add_argument('-s', '--start-time', dest="startTime", default='00:00:00', type=str)
	parser.add_argument('-d', '--duration', dest="duration", default='00:00:00', type=str)
	return parser.parse_args()

def main():
	option = parseArg()
	if not os.path.isfile(option.videoFile):
		return
	startTime =(int(option.startTime.split(':')[0])*60+int(option.startTime.split(':')[1]))*60+int(option.startTime.split(':')[2])
	duration =(int(option.duration.split(':')[0])*60+int(option.duration.split(':')[1]))*60+int(option.duration.split(':')[2])

	cuttingVideo(option.videoFile, startTime, duration)

if __name__ == '__main__':
	main()