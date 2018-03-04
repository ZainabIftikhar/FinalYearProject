import cv2
import sys

import pickle

import numpy as np

from ModelGeneration.neuralnetworkmodel import NeuralNetworkModel




neuralModel = NeuralNetworkModel()

neuralModelPickle = open("ModelStorage/nnlivetest.pickle", "rb")

neuralModelDict = pickle.load(neuralModelPickle)

video_capture = cv2.VideoCapture(0)

frame_count = 0
openness_average = []
openness_min = 1
openness_max = 0

extraversion_average = []
extraversion_min = 1
extraversion_max = 0

neuroticism_average = []
neuroticism_min = 1
neuroticism_max = 0

agreeableness_average = []
agreeableness_min = 1
agreeableness_max = 0

conscientiousness_average = []
conscientiousness_min = 1
conscientiousness_max = 0
while True:
    ret, frame = video_capture.read()

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)


    if frame_count % 15 is 0:
        temp_openness, temp_extraversion, temp_neuroticism, temp_agreeableness, temp_conscientiousness = neuralModel.predict_single_frame(frame, neuralModelDict)

        if temp_openness["average"] is not -1:
            openness_average.append(temp_openness["average"])
            if temp_openness["min"] < openness_min:
                openness_min = temp_openness["min"]
            if temp_openness["max"] > openness_max:
                openness_max = temp_openness["max"]
        if temp_extraversion["average"] is not -1:
            extraversion_average.append(temp_extraversion["average"])
            if temp_extraversion["min"] < extraversion_min:
                extraversion_min = temp_extraversion["min"]
            if temp_extraversion["max"] > extraversion_max:
                extraversion_max = temp_extraversion["max"]
        if temp_agreeableness["average"] is not -1:
            agreeableness_average.append(temp_agreeableness["average"])
            if temp_agreeableness["min"] < agreeableness_min:
                agreeableness_min = temp_agreeableness["min"]
            if temp_agreeableness["max"] > agreeableness_max:
                agreeableness_max = temp_agreeableness["max"]
        if temp_conscientiousness["average"] is not -1:
            conscientiousness_average.append(temp_conscientiousness["average"])
            if temp_conscientiousness["min"] < conscientiousness_min:
                conscientiousness_min = temp_conscientiousness["min"]
            if temp_conscientiousness["max"] > conscientiousness_max:
                conscientiousness_max = temp_conscientiousness["max"]
        if temp_neuroticism["average"] is not -1:
            neuroticism_average.append(temp_neuroticism["average"])
            if temp_neuroticism["min"] < neuroticism_min:
                neuroticism_min = temp_neuroticism["min"]
            if temp_neuroticism["max"] > neuroticism_max:
                neuroticism_max = temp_neuroticism["max"]
        font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "Openness (Avg) = {0:.4f}".format(np.average(openness_average)), (10, 20), font, 0.45, color)
    cv2.putText(frame, "Extraversion (Avg) = {0:.4f}".format(np.average(extraversion_average)), (10, 40), font, 0.45, color)
    cv2.putText(frame, "Neuroticism (Avg) = {0:.4f}".format(np.average(neuroticism_average)), (10, 60), font, 0.45, color)
    cv2.putText(frame, "Agreeableness (Avg) = {0:.4f}".format(np.average(agreeableness_average)), (10, 80), font, 0.45, color)
    cv2.putText(frame, "Consentiousness (Avg)= {0:.4f}".format(np.average(conscientiousness_average)), (10, 100), font, 0.45, color)


    cv2.putText(frame, "Openness (Min) = {0:.4f}".format(openness_min), (10, 140), font, 0.45, color)
    cv2.putText(frame, "Extraversion (Min) = {0:.4f}".format(extraversion_min), (10, 160), font, 0.45, color)
    cv2.putText(frame, "Neuroticism (Min) = {0:.4f}".format(neuroticism_min), (10, 180), font, 0.45, color)
    cv2.putText(frame, "Agreeableness (Min) = {0:.4f}".format(agreeableness_min), (10, 200), font, 0.45, color)
    cv2.putText(frame, "Consentiousness (Min)= {0:.4f}".format(conscientiousness_min), (10, 220), font, 0.45, color)

    cv2.putText(frame, "Openness (Max) = {0:.4f}".format(openness_max), (10, 260), font, 0.45, color)
    cv2.putText(frame, "Extraversion (Max) = {0:.4f}".format(extraversion_max), (10, 280), font, 0.45, color)
    cv2.putText(frame, "Neuroticism (Max) = {0:.4f}".format(neuroticism_max), (10, 300), font, 0.45, color)
    cv2.putText(frame, "Agreeableness (Max) = {0:.4f}".format(agreeableness_max), (10, 320), font, 0.45, color)
    cv2.putText(frame, "Consentiousness (Max)= {0:.4f}".format(conscientiousness_max), (10, 340), font, 0.45, color)

    frame_count+=1
    cv2.imshow("output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()