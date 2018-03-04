import pickle
import subprocess

videosFilePath = 'ValidationVideos/'
audioFilePath = 'ValidationVideosAudio/'

videosDataFile = open("AnnotationFiles/annotation_validation.pkl", "rb")
print('Loading data from pickle file.')
videosData = pickle.load(videosDataFile, encoding='latin1')



print('Getting names of all the video files.')
videoNames = list(videosData['extraversion'].keys())

i = 1
for videoName in videoNames:
    position = videoName.find(".mp4")
    audioFileName = audioFilePath + videoName[:position] + '.wav'

    print(audioFileName)

    command = "ffmpeg -i C:/Users/shaoo/PycharmProjects/FinalYearProject/ValidationVideos/{} -ab 160k -ac 2 -ar 44100 -vn {}".format(videoName, audioFileName)
    subprocess.call(command, shell=True)
    print("Processing Video No: {}".format(i))
    i=i+1