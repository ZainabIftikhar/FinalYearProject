import pickle
import random
from collections import defaultdict

import numpy as np
import xlsxwriter
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

            NNModelDict = defaultdict(dict)

            NNModelDict['openness'] = opennessModelEmotion
            NNModelDict['agreeableness'] = agreeablenessModelEmotion
            NNModelDict['extraversion'] = extraversionModelEmotion
            NNModelDict['conscientiousness'] = conscientiousnessModelEmotion
            NNModelDict['neuroticism'] = neuroticismModelEmotion

            NNDataFile = open("../AnnotationFiles/emotionnn.pkl", "wb")
            pickle.dump(NNModelDict, NNDataFile)

    def test (self):
        self.modelDict = pickle.load(open("../ModelStorage/emotionnn.pkl", "rb"))


        TestEmotionDictPickle = open("testdict.pickle", "rb")

        TestEmotionDict = pickle.load(TestEmotionDictPickle)

        workbook = xlsxwriter.Workbook("emotionnndata.xlsx", {'nan_inf_to_errors': True})

        worksheet = workbook.add_worksheet()

        bold = workbook.add_format({'bold': True})

        worksheet.write('A1', 'Openness (Expected)', bold)
        worksheet.write('B1', 'Openness (Actual)', bold)
        worksheet.write('C1', 'Openness (Error)', bold)

        worksheet.write('D1', 'Extraversion (Expected)', bold)
        worksheet.write('E1', 'Extraversion (Actual)', bold)
        worksheet.write('F1', 'Extraversion (Error)', bold)

        worksheet.write('G1', 'Neuroticism (Expected)', bold)
        worksheet.write('H1', 'Neuroticism (Actual)', bold)
        worksheet.write('I1', 'Neuroticism (Error)', bold)

        worksheet.write('J1', 'Agreeableness (Expected)', bold)
        worksheet.write('K1', 'Agreeableness (Actual)', bold)
        worksheet.write('L1', 'Agreeableness (Error)', bold)

        worksheet.write('M1', 'Conscientiousness (Expected)', bold)
        worksheet.write('N1', 'Conscientiousness (Actual)', bold)
        worksheet.write('O1', 'Conscientiousness (Error)', bold)

        videosDataFile = open("../AnnotationFiles/annotation_test.pkl", "rb")
        print('Loading data from pickle file.')


        videosData = pickle.load(videosDataFile, encoding='latin1')

        print('Getting names of all the video files.')
        videoNames = list(videosData['extraversion'].keys())

        videosFilePath = 'TestVideos/'

        opennessMSE = []
        extraversionMSE = []
        neuroticismMSE = []
        conscientiousnessMSE = []
        agreeablenessMSE = []

        for i, videoName in enumerate(videoNames):

            openness = 0

            extraversion = 0

            neuroticism = 0

            agreeableness = 0

            conscientiousness = 0

            print("Video number: {}".format(i + 1))

            position = videoName.find(".mp4")
            audioFileName = videoName[:position] + '.wav'

            if audioFileName in TestEmotionDict:

                feature = []

                feature.append(TestEmotionDict[audioFileName]['neutral'])

                feature.append(TestEmotionDict[audioFileName]['happy'])

                feature.append(TestEmotionDict[audioFileName]['sad'])

                feature.append(TestEmotionDict[audioFileName]['angry'])

                feature.append(TestEmotionDict[audioFileName]['fear'])

                features = np.array(feature)
                features = features.reshape(1, -1)

                openness = self.modelDict['openness'].predict(features)

                extraversion = self.modelDict['extraversion'].predict(features)

                neuroticism = self.modelDict['neuroticism'].predict(features)

                agreeableness = self.modelDict['agreeableness'].predict(features)

                conscientiousness = self.modelDict['conscientiousness'].predict(features)

                worksheet.write(i + 1, 0, videosData['openness'][videoName])

                worksheet.write(i + 1, 1, openness)
                worksheet.write(i + 1, 2, np.square(videosData['openness'][videoName] - openness))
                opennessMSE.append(np.square(videosData['openness'][videoName] - openness))

                worksheet.write(i + 1, 3, videosData['extraversion'][videoName])

                worksheet.write(i + 1, 4, extraversion)
                worksheet.write(i + 1, 5, np.square(videosData['extraversion'][videoName] - extraversion))
                extraversionMSE.append(np.square(videosData['extraversion'][videoName] - extraversion))

                worksheet.write(i + 1, 6, videosData['neuroticism'][videoName])

                worksheet.write(i + 1, 7, neuroticism)
                worksheet.write(i + 1, 8, np.square(videosData['neuroticism'][videoName] - neuroticism))
                neuroticismMSE.append(np.square(videosData['neuroticism'][videoName] - neuroticism))

                worksheet.write(i + 1, 9, videosData['agreeableness'][videoName])

                worksheet.write(i + 1, 10, agreeableness)
                worksheet.write(i + 1, 11, np.square(videosData['agreeableness'][videoName] - agreeableness))
                agreeablenessMSE.append(np.square(videosData['agreeableness'][videoName] - agreeableness))

                worksheet.write(i + 1, 12, videosData['conscientiousness'][videoName])

                worksheet.write(i + 1, 13, conscientiousness)
                worksheet.write(i + 1, 14, np.square(videosData['conscientiousness'][videoName] - conscientiousness))
                conscientiousnessMSE.append(np.square(videosData['conscientiousness'][videoName] - conscientiousness))

        workbook.close()

        ErrorDict = defaultdict(float)

        ErrorDict['openness'] = np.sqrt(np.average(opennessMSE))
        ErrorDict['agreeableness'] = np.sqrt(np.average(agreeablenessMSE))
        ErrorDict['neuroticism'] = np.sqrt(np.average(neuroticismMSE))
        ErrorDict['conscientiousness'] = np.sqrt(np.average(conscientiousnessMSE))
        ErrorDict['extraversion'] = np.sqrt(np.average(extraversionMSE))

        return ErrorDict

NeuralNetwork = NeuralNetworkModelEmotion()

params = {
    'hidden_layer_sizes': (190, 190, 190, 190,),
    'activation': 'relu',
    'solver': 'adam',
    'learning_rate': 'adaptive',
    'alpha': 0.00001
}

NeuralNetwork.test()

