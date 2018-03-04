# OpenVokaWavMean-win64.py
# public-domain sample code by Vokaturi, 2018-02-20
#
# A sample script that uses the VokaturiPlus library to extract the emotions from
# a wav file on disk. The file has to contain a mono recording.
#
# Call syntax:
#   python3 OpenVokaWavMean-win64.py path_to_sound_file.wav
#
# For the sound file hello.wav that comes with OpenVokaturi, the result should be:
#	Neutral: 0.760
#	Happy: 0.000
#	Sad: 0.238
#	Angry: 0.001
#	Fear: 0.000

import sys

import os
import scipy.io.wavfile
import xlsxwriter as xlsxwriter

sys.path.append("../api")
import Vokaturi

print ("Loading library...")
Vokaturi.load("../OpenVokaturi-3-0-win64.dll")
print ("Analyzed by: %s" % Vokaturi.versionAndLicense())

print ("Reading sound file...")

workbook = xlsxwriter.Workbook("Output.xlsx", {'nan_inf_to_errors': True})

worksheet = workbook.add_worksheet()

bold = workbook.add_format({'bold': True})

worksheet.write('A1', 'Filename', bold)
worksheet.write('B1', 'Neutral', bold)
worksheet.write('C1', 'Happy', bold)
worksheet.write('D1', 'Sad', bold)
worksheet.write('E1', 'Angry', bold)
worksheet.write('F1', 'Fear', bold)

i = 1

for filename in os.listdir(os.getcwd() + "/Messages"):
    print(filename)

    worksheet.write(i, 0, filename)

    (sample_rate, samples) = scipy.io.wavfile.read("Messages/"+filename)
    print ("   sample rate %.3f Hz" % sample_rate)

    print ("Allocating Vokaturi sample array...")
    buffer_length = len(samples)
    print ("   %d samples, %d channels" % (buffer_length, samples.ndim))
    c_buffer = Vokaturi.SampleArrayC(buffer_length)
    if samples.ndim == 1:  # mono
        c_buffer[:] = samples[:] / 32768.0
    else:  # stereo
        c_buffer[:] = 0.5*(samples[:,0]+0.0+samples[:,1]) / 32768.0

    print ("Creating VokaturiVoice...")
    voice = Vokaturi.Voice (sample_rate, buffer_length)

    print ("Filling VokaturiVoice with samples...")
    voice.fill(buffer_length, c_buffer)

    print ("Extracting emotions from VokaturiVoice...")
    quality = Vokaturi.Quality()
    emotionProbabilities = Vokaturi.EmotionProbabilities()
    voice.extract(quality, emotionProbabilities)

    if quality.valid:
        print ("Neutral: %.3f" % emotionProbabilities.neutrality)
        worksheet.write(i, 1, emotionProbabilities.neutrality)

        print ("Happy: %.3f" % emotionProbabilities.happiness)
        worksheet.write(i, 2, emotionProbabilities.happiness)

        print ("Sad: %.3f" % emotionProbabilities.sadness)
        worksheet.write(i, 3, emotionProbabilities.sadness)

        print ("Angry: %.3f" % emotionProbabilities.anger)
        worksheet.write(i, 4, emotionProbabilities.anger)

        print ("Fear: %.3f" % emotionProbabilities.fear)
        worksheet.write(i, 5, emotionProbabilities.fear)

    else:
        print ("Not enough sonorancy to determine emotions")

    voice.destroy()
    i=i+1

workbook.close()
