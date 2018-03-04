import pickle
import random

import numpy as np
from sklearn.neural_network import MLPRegressor

class NeuralNetworkModelEmotion:
    def __init__(self, featuresDictTrain=None):
        self.featuresDictTrain = featuresDictTrain

    def generate(self, params):

        neuroticismModelEmotion = MLPRegressor().set_params(**params)
        extraversionModelEmotion = MLPRegressor().set_params(**params)
        conscientiousnessModelEmotion = MLPRegressor().set_params(**params)
        agreeablenessModelEmotion = MLPRegressor().set_params(**params)
        opennessModelEmotion = MLPRegressor().set_params(**params)

        print("Training Started")

        TrainingEmotionDictPickle = open("trainingdict.pickle", "rb")

        TrainingEmotionDict = pickle.load(TrainingEmotionDictPickle)

        videosDataFile = open("../AnnotationFiles/annotation_training.pkl", "rb")
        print('Loading data from pickle file.')

        videosData = pickle.load(videosDataFile, encoding='latin1')

        print('Getting names of all the video files.')
        videoNames = list(videosData['extraversion'].keys())

        i = 1

        for videoName in videoNames:

            print(videoName)

            position = videoName.find(".mp4")
            audioFileName = videoName[:position] + '.wav'

            print (audioFileName)

            if audioFileName in TrainingEmotionDict:

                feature = []

                feature.append(TrainingEmotionDict[audioFileName]['neutral'])

                feature.append(TrainingEmotionDict[audioFileName]['happy'])

                feature.append(TrainingEmotionDict[audioFileName]['sad'])

                feature.append(TrainingEmotionDict[audioFileName]['angry'])

                feature.append(TrainingEmotionDict[audioFileName]['fear'])

                features = np.array(feature)
                features = features.reshape(1, -1)

                neuroticismModelEmotion.fit(features, np.array(videosData['neuroticism'][videoName]).ravel())

                extraversionModelEmotion.fit(features, np.array(videosData['extraversion'][videoName]).ravel())

                opennessModelEmotion.fit(features, np.array(videosData['openness'][videoName]).ravel())

                agreeablenessModelEmotion.fit(features, np.array(videosData['agreeableness'][videoName]).ravel())

                conscientiousnessModelEmotion.fit(features, np.array(videosData['conscientiousness'][videoName]).ravel())

                print("File number: {}".format(i))
                i = i+1


        for k in range (1, 3):
            randomfilename = random.choice(videoNames)

            position = videoName.find(".mp4")
            audioFileName = videoName[:position] + '.wav'

            feature = []

            feature.append(TrainingEmotionDict[audioFileName]['neutral'])

            feature.append(TrainingEmotionDict[audioFileName]['happy'])

            feature.append(TrainingEmotionDict[audioFileName]['sad'])

            feature.append(TrainingEmotionDict[audioFileName]['angry'])

            feature.append(TrainingEmotionDict[audioFileName]['fear'])

            features = np.array(feature)
            features = features.reshape(1, -1)

            print("The prediction for openness of file {} is: {} ".format(randomfilename, opennessModelEmotion.predict(features)))
            print("The actual value is: {} ".format(videosData['openness'][randomfilename]))


            print("The prediction for agreeableness of file {} is: {} ".format(randomfilename, agreeablenessModelEmotion.predict(features)))
            print("The actual value is: {} ".format(videosData['agreeableness'][randomfilename]))


            print("The prediction for neuroticism of file {} is: {} ".format(randomfilename, neuroticismModelEmotion.predict(features)))
            print("The actual value is: {} ".format(videosData['neuroticism'][randomfilename]))


            print("The prediction for extraversion of file {} is: {} ".format(randomfilename, extraversionModelEmotion.predict(features)))
            print("The actual value is: {} ".format(videosData['extraversion'][randomfilename]))


            print("The prediction for conscientiousness of file {} is: {} ".format(randomfilename, conscientiousnessModelEmotion.predict(features)))
            print("The actual value is: {} ".format(videosData['conscientiousness'][randomfilename]))

NeuralNetwork = NeuralNetworkModelEmotion()

params = {
    'hidden_layer_sizes': (190, 190, 190, 190,),
    'activation': 'relu',
    'solver': 'adam',
    'learning_rate': 'adaptive',
    'alpha': 0.00001
}

NeuralNetwork.generate(params)