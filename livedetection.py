import cv2
import sys

import pickle

from ModelGeneration.neuralnetworkmodel import NeuralNetworkModel




neuralModel = NeuralNetworkModel()

neuralModelPickle = open("ModelStorage/nnadam.pickle", "rb")

neuralModelDict = pickle.load(neuralModelPickle)

video_capture = cv2.VideoCapture(0)

frame_count = 0
openness = 0
extraversion = 0
neuroticism = 0
agreeableness = 0
conscientiousness = 0
while True:
    ret, frame = video_capture.read()

    font = cv2.FONT_HERSHEY_SIMPLEX


    if frame_count % 15 is 0:
        temp_openness, temp_extraversion, temp_neuroticism, temp_agreeableness, temp_conscientiousness = neuralModel.predict_single_frame(frame, neuralModelDict)

        if temp_openness is not -1:
            openness = temp_openness
        if temp_extraversion is not -1:
            extraversion = temp_extraversion
        if temp_agreeableness is not -1:
            agreeableness = temp_agreeableness
        if temp_conscientiousness is not -1:
            conscientiousness = temp_conscientiousness
        if temp_neuroticism is not -1:
            neuroticism = temp_neuroticism
        font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "Openness = {}".format(openness), (10, 15), font, 0.5, 255)
    cv2.putText(frame, "Extraversion = {}".format(extraversion), (10, 35), font, 0.5, 255)
    cv2.putText(frame, "Neuroticism = {}".format(neuroticism), (10, 55), font, 0.5, 255)
    cv2.putText(frame, "Agreeableness = {}".format(agreeableness), (10, 75), font, 0.5, 255)
    cv2.putText(frame, "Consentiousness = {}".format(conscientiousness), (10, 95), font, 0.5, 255)

    frame_count+=1
    cv2.imshow("output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()