from collections import defaultdict

import pickle

import numpy as np
import time
import xlsxwriter
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from VideoProcessing.featureextraction import FeatureExtraction
from VideoProcessing.frameextraction import FrameExtraction
from matplotlib import pyplot as plt


class NeuralNetworkModel:
    def __init__(self, featuresDictTrain=None):
        self.featuresDictTrain = featuresDictTrain

    def find_best_params(self, featureDictTrain, featureDictTest):

        opennessModel = MLPRegressor(hidden_layer_sizes=(15, 15, 15,), activation='tanh', solver='adam',
                                     learning_rate='adaptive')

        print("Fitting the model")

        opennessModel.fit(featureDictTrain['face']['hist'], featureDictTrain['face']['openness'])

        print("Getting the prediction")

        prediction = opennessModel.predict(featureDictTest['face']['hist'])

        plt.scatter(featureDictTest['face']['openness'], prediction)
        plt.xlabel('TrueValues')
        plt.ylabel('Predictions')
        plt.show()

        print("Score: {}".format(
            opennessModel.score(featureDictTest['face']['hist'], featureDictTest['face']['openness'])))

    def generate(self, params):
        neuroticismModelRightEye = MLPRegressor().set_params(**params)
        neuroticismModelLeftEye = MLPRegressor().set_params(**params)
        neuroticismModelFace = MLPRegressor().set_params(**params)
        neuroticismModelSmile = MLPRegressor().set_params(**params)

        extraversionModelRightEye = MLPRegressor().set_params(**params)
        extraversionModelLeftEye = MLPRegressor().set_params(**params)
        extraversionModelFace = MLPRegressor().set_params(**params)
        extraversionModelSmile = MLPRegressor().set_params(**params)

        conscientiousnessModelRightEye = MLPRegressor().set_params(**params)
        conscientiousnessModelLeftEye = MLPRegressor().set_params(**params)
        conscientiousnessModelFace = MLPRegressor().set_params(**params)
        conscientiousnessModelSmile = MLPRegressor().set_params(**params)

        agreeablenessModelRightEye = MLPRegressor().set_params(**params)
        agreeablenessModelLeftEye = MLPRegressor().set_params(**params)
        agreeablenessModelFace = MLPRegressor().set_params(**params)
        agreeablenessModelSmile = MLPRegressor().set_params(**params)

        opennessModelRightEye = MLPRegressor().set_params(**params)
        opennessModelLeftEye = MLPRegressor().set_params(**params)
        opennessModelFace = MLPRegressor().set_params(**params)
        opennessModelSmile = MLPRegressor().set_params(**params)

        print("Training started")

        neuroticismModelRightEye.fit(self.featuresDictTrain['righteye']['hist'],
                                     self.featuresDictTrain['righteye']['neuroticism'])
        neuroticismModelLeftEye.fit(self.featuresDictTrain['lefteye']['hist'],
                                    self.featuresDictTrain['lefteye']['neuroticism'])
        neuroticismModelFace.fit(self.featuresDictTrain['face']['hist'], self.featuresDictTrain['face']['neuroticism'])
        neuroticismModelSmile.fit(self.featuresDictTrain['smile']['hist'],
                                  self.featuresDictTrain['smile']['neuroticism'])

        opennessModelRightEye.fit(self.featuresDictTrain['righteye']['hist'],
                                  self.featuresDictTrain['righteye']['openness'])
        opennessModelLeftEye.fit(self.featuresDictTrain['lefteye']['hist'],
                                 self.featuresDictTrain['lefteye']['openness'])
        opennessModelFace.fit(self.featuresDictTrain['face']['hist'], self.featuresDictTrain['face']['openness'])
        opennessModelSmile.fit(self.featuresDictTrain['smile']['hist'], self.featuresDictTrain['smile']['openness'])

        agreeablenessModelRightEye.fit(self.featuresDictTrain['righteye']['hist'],
                                       self.featuresDictTrain['righteye']['agreeableness'])
        agreeablenessModelLeftEye.fit(self.featuresDictTrain['lefteye']['hist'],
                                      self.featuresDictTrain['lefteye']['agreeableness'])
        agreeablenessModelFace.fit(self.featuresDictTrain['face']['hist'],
                                   self.featuresDictTrain['face']['agreeableness'])
        agreeablenessModelSmile.fit(self.featuresDictTrain['smile']['hist'],
                                    self.featuresDictTrain['smile']['agreeableness'])

        extraversionModelRightEye.fit(self.featuresDictTrain['righteye']['hist'],
                                      self.featuresDictTrain['righteye']['extraversion'])
        extraversionModelLeftEye.fit(self.featuresDictTrain['lefteye']['hist'],
                                     self.featuresDictTrain['lefteye']['extraversion'])
        extraversionModelFace.fit(self.featuresDictTrain['face']['hist'],
                                  self.featuresDictTrain['face']['extraversion'])
        extraversionModelSmile.fit(self.featuresDictTrain['smile']['hist'],
                                   self.featuresDictTrain['smile']['extraversion'])

        conscientiousnessModelRightEye.fit(self.featuresDictTrain['righteye']['hist'],
                                           self.featuresDictTrain['righteye']['conscientiousness'])
        conscientiousnessModelLeftEye.fit(self.featuresDictTrain['lefteye']['hist'],
                                          self.featuresDictTrain['lefteye']['conscientiousness'])
        conscientiousnessModelFace.fit(self.featuresDictTrain['face']['hist'],
                                       self.featuresDictTrain['face']['conscientiousness'])
        conscientiousnessModelSmile.fit(self.featuresDictTrain['smile']['hist'],
                                        self.featuresDictTrain['smile']['conscientiousness'])

        print("Training done")

        NNModelDict = defaultdict(dict)

        NNModelDict['face']['openness'] = opennessModelFace
        NNModelDict['face']['agreeableness'] = agreeablenessModelFace
        NNModelDict['face']['extraversion'] = extraversionModelFace
        NNModelDict['face']['conscientiousness'] = conscientiousnessModelFace
        NNModelDict['face']['neuroticism'] = neuroticismModelFace

        NNModelDict['smile']['openness'] = opennessModelSmile
        NNModelDict['smile']['agreeableness'] = agreeablenessModelSmile
        NNModelDict['smile']['extraversion'] = extraversionModelSmile
        NNModelDict['smile']['conscientiousness'] = conscientiousnessModelSmile
        NNModelDict['smile']['neuroticism'] = neuroticismModelSmile

        NNModelDict['righteye']['openness'] = opennessModelRightEye
        NNModelDict['righteye']['agreeableness'] = agreeablenessModelRightEye
        NNModelDict['righteye']['extraversion'] = extraversionModelRightEye
        NNModelDict['righteye']['conscientiousness'] = conscientiousnessModelRightEye
        NNModelDict['righteye']['neuroticism'] = neuroticismModelRightEye

        NNModelDict['lefteye']['openness'] = opennessModelLeftEye
        NNModelDict['lefteye']['agreeableness'] = agreeablenessModelLeftEye
        NNModelDict['lefteye']['extraversion'] = extraversionModelLeftEye
        NNModelDict['lefteye']['conscientiousness'] = conscientiousnessModelLeftEye
        NNModelDict['lefteye']['neuroticism'] = neuroticismModelLeftEye

        self.modelDict = NNModelDict
        return NNModelDict

    def test(self, facesHist, smileHist, leftEyeHist, rightEyeHist, filename, model=None, videoCount=1000):

        if model:
            self.modelDict = model

        workbook = xlsxwriter.Workbook(filename, {'nan_inf_to_errors': True})

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

        videosDataFile = open("AnnotationFiles/annotation_test.pkl", "rb")
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

            opennessListFace = []
            opennessListLeftEye = []
            opennessListRightEye = []
            opennessListSmile = []

            extraversionListFace = []
            extraversionListLeftEye = []
            extraversionListRightEye = []
            extraversionListSmile = []

            neuroticismListFace = []
            neuroticismListLeftEye = []
            neuroticismListRightEye = []
            neuroticismListSmile = []

            agreeablenessListFace = []
            agreeablenessListLeftEye = []
            agreeablenessListRightEye = []
            agreeablenessListSmile = []

            conscientiousnessListFace = []
            conscientiousnessListLeftEye = []
            conscientiousnessListRightEye = []
            conscientiousnessListSmile = []

            print("Video number: {}".format(i + 1))

            if facesHist[videoName]:
                opennessListFace = self.modelDict['face']['openness'].predict(facesHist[videoName])
                extraversionListFace = self.modelDict['face']['extraversion'].predict(facesHist[videoName])
                neuroticismListFace = self.modelDict['face']['neuroticism'].predict(facesHist[videoName])
                agreeablenessListFace = self.modelDict['face']['agreeableness'].predict(facesHist[videoName])
                conscientiousnessListFace = self.modelDict['face']['conscientiousness'].predict(facesHist[videoName])

            if rightEyeHist[videoName]:
                opennessListRightEye = self.modelDict['righteye']['openness'].predict(rightEyeHist[videoName])
                extraversionListRightEye = self.modelDict['righteye']['extraversion'].predict(rightEyeHist[videoName])
                neuroticismListRightEye = self.modelDict['righteye']['neuroticism'].predict(rightEyeHist[videoName])
                agreeablenessListRightEye = self.modelDict['righteye']['agreeableness'].predict(rightEyeHist[videoName])
                conscientiousnessListRightEye = self.modelDict['righteye']['conscientiousness'].predict(
                    rightEyeHist[videoName])

            if leftEyeHist[videoName]:
                opennessListLeftEye = self.modelDict['lefteye']['openness'].predict(leftEyeHist[videoName])
                extraversionListLeftEye = self.modelDict['lefteye']['extraversion'].predict(leftEyeHist[videoName])
                neuroticismListLeftEye = self.modelDict['lefteye']['neuroticism'].predict(leftEyeHist[videoName])
                agreeablenessListLeftEye = self.modelDict['lefteye']['agreeableness'].predict(leftEyeHist[videoName])
                conscientiousnessListLeftEye = self.modelDict['lefteye']['conscientiousness'].predict(
                    leftEyeHist[videoName])

            if smileHist[videoName]:
                opennessListSmile = self.modelDict['smile']['openness'].predict(smileHist[videoName])
                extraversionListSmile = self.modelDict['smile']['extraversion'].predict(smileHist[videoName])
                neuroticismListSmile = self.modelDict['smile']['neuroticism'].predict(smileHist[videoName])
                agreeablenessListSmile = self.modelDict['smile']['agreeableness'].predict(smileHist[videoName])
                conscientiousnessListSmile = self.modelDict['smile']['conscientiousness'].predict(smileHist[videoName])

            tempList = np.concatenate((opennessListFace, opennessListLeftEye, opennessListRightEye, opennessListSmile))


            worksheet.write(i + 1, 0, videosData['openness'][videoName])
            # worksheet.write(i + 1, 1, sum(opennessListFace) / len(opennessListFace))
            # worksheet.write(i + 1, 2, sum(opennessListLeftEye) / len(opennessListLeftEye))
            # worksheet.write(i + 1, 3, sum(opennessListRightEye) / len(opennessListRightEye))
            # worksheet.write(i + 1, 4, sum(opennessListSmile) / len(opennessListSmile))
            if len(tempList) is not 0:
                worksheet.write(i + 1, 1, np.average(tempList))
                worksheet.write(i + 1, 2, np.square(videosData['openness'][videoName] - np.average(tempList)))
                opennessMSE.append(np.square(videosData['openness'][videoName] - np.average(tempList)))

            tempList = np.concatenate(
                (extraversionListFace, extraversionListLeftEye, extraversionListRightEye, extraversionListSmile))

            worksheet.write(i + 1, 3, videosData['extraversion'][videoName])
            # worksheet.write(i + 1, 7, sum(extraversionListFace) / len(extraversionListFace))
            # worksheet.write(i + 1, 8, sum(extraversionListLeftEye) / len(extraversionListLeftEye))
            # worksheet.write(i + 1, 9, sum(extraversionListRightEye) / len(extraversionListRightEye))
            # worksheet.write(i + 1, 10, sum(extraversionListSmile) / len(extraversionListSmile))
            if len(tempList) is not 0:
                worksheet.write(i + 1, 4, np.max(tempList))
                worksheet.write(i + 1, 5, np.square(videosData['extraversion'][videoName] - np.average(tempList)))
                extraversionMSE.append(np.square(videosData['extraversion'][videoName] - np.average(tempList)))

            tempList = np.concatenate(
                (neuroticismListFace, neuroticismListLeftEye, neuroticismListRightEye, neuroticismListSmile))

            worksheet.write(i + 1, 6, videosData['neuroticism'][videoName])
            # worksheet.write(i + 1, 13, sum(neuroticismListFace) / len(neuroticismListFace))
            # worksheet.write(i + 1, 14, sum(neuroticismListLeftEye) / len(neuroticismListLeftEye))
            # worksheet.write(i + 1, 15, sum(neuroticismListRightEye) / len(neuroticismListRightEye))
            # worksheet.write(i + 1, 16, sum(neuroticismListSmile) / len(neuroticismListSmile))
            if len(tempList) is not 0:
                worksheet.write(i + 1, 7, np.max(tempList))
                worksheet.write(i + 1, 8, np.square(videosData['neuroticism'][videoName] - np.average(tempList)))
                neuroticismMSE.append(np.square(videosData['neuroticism'][videoName] - np.average(tempList)))

            tempList = np.concatenate(
                (agreeablenessListFace, agreeablenessListLeftEye, agreeablenessListRightEye, agreeablenessListSmile))

            worksheet.write(i + 1, 9, videosData['agreeableness'][videoName])
            # worksheet.write(i + 1, 19, sum(agreeablenessListFace) / len(agreeablenessListFace))
            # worksheet.write(i + 1, 20, sum(agreeablenessListLeftEye) / len(agreeablenessListLeftEye))
            # worksheet.write(i + 1, 21, sum(agreeablenessListRightEye) / len(agreeablenessListRightEye))
            # worksheet.write(i + 1, 22, sum(agreeablenessListSmile) / len(agreeablenessListSmile))
            if len(tempList) is not 0:
                worksheet.write(i + 1, 10, np.max(tempList))
                worksheet.write(i + 1, 11, np.square(videosData['agreeableness'][videoName] - np.average(tempList)))
                agreeablenessMSE.append(np.square(videosData['agreeableness'][videoName] - np.average(tempList)))

            tempList = np.concatenate((conscientiousnessListFace, conscientiousnessListLeftEye,
                                       conscientiousnessListRightEye, conscientiousnessListSmile))

            worksheet.write(i + 1, 12, videosData['conscientiousness'][videoName])
            # worksheet.write(i + 1, 25, sum(conscientiousnessListFace) / len(conscientiousnessListFace))
            # worksheet.write(i + 1, 26, sum(conscientiousnessListLeftEye) / len(conscientiousnessListLeftEye))
            # worksheet.write(i + 1, 27, sum(conscientiousnessListRightEye) / len(conscientiousnessListRightEye))
            # worksheet.write(i + 1, 28, sum(conscientiousnessListSmile) / len(conscientiousnessListSmile))
            if len(tempList) is not 0:
                worksheet.write(i + 1, 13, np.max(tempList))
                worksheet.write(i + 1, 14, np.square(videosData['conscientiousness'][videoName] - np.average(tempList)))
                conscientiousnessMSE.append(np.square(videosData['conscientiousness'][videoName] - np.average(tempList)))


        workbook.close()

        ErrorDict = defaultdict(float)


        ErrorDict['openness'] = np.sqrt(np.average(opennessMSE))
        ErrorDict['agreeableness'] = np.sqrt(np.average(agreeablenessMSE))
        ErrorDict['neuroticism'] = np.sqrt(np.average(neuroticismMSE))
        ErrorDict['conscientiousness'] = np.sqrt(np.average(conscientiousnessMSE))
        ErrorDict['extraversion'] = np.sqrt(np.average(extraversionMSE))

        return ErrorDict


    def predict_single_frame(self, frame, model=None):
        if model is not None:
            self.modelDict = model

        print("Trying to predict a single frame value")

        neuroticismListFace = []
        neuroticismListLeftEye = []
        neuroticismListRightEye = []
        agreeablenessListFace = []
        agreeablenessListLeftEye = []
        neuroticismListSmile = []
        agreeablenessListSmile = []
        agreeablenessListRightEye = []
        conscientiousnessListFace = []
        conscientiousnessListLeftEye = []
        conscientiousnessListRightEye = []
        conscientiousnessListSmile = []
        extraversionListFace = []
        extraversionListLeftEye = []
        extraversionListRightEye = []
        extraversionListSmile = []
        opennessListFace = []
        opennessListLeftEye = []
        opennessListRightEye = []
        opennessListSmile = []

        extractor = FrameExtraction()
        features = FeatureExtraction(24, 8)

        croppedFace, smileFrame, leftEyeFrame, rightEyeFrame = extractor.single_frame(frame)
        featuresDict = features.extract_single_frame(croppedFace, smileFrame, leftEyeFrame, rightEyeFrame)

        if featuresDict['face']:
            opennessListFace = self.modelDict['face']['openness'].predict(featuresDict['face'])
            extraversionListFace = self.modelDict['face']['extraversion'].predict(featuresDict['face'])
            neuroticismListFace = self.modelDict['face']['neuroticism'].predict(featuresDict['face'])
            agreeablenessListFace = self.modelDict['face']['agreeableness'].predict(featuresDict['face'])
            conscientiousnessListFace = self.modelDict['face']['conscientiousness'].predict(featuresDict['face'])

        if featuresDict['righteye']:
            opennessListRightEye = self.modelDict['righteye']['openness'].predict(featuresDict['righteye'])
            extraversionListRightEye = self.modelDict['righteye']['extraversion'].predict(featuresDict['righteye'])
            neuroticismListRightEye = self.modelDict['righteye']['neuroticism'].predict(featuresDict['righteye'])
            agreeablenessListRightEye = self.modelDict['righteye']['agreeableness'].predict(
                featuresDict['righteye'])
            conscientiousnessListRightEye = self.modelDict['righteye']['conscientiousness'].predict(
                featuresDict['righteye'])

        if featuresDict['lefteye']:
            opennessListLeftEye = self.modelDict['lefteye']['openness'].predict(featuresDict['lefteye'])
            extraversionListLeftEye = self.modelDict['lefteye']['extraversion'].predict(featuresDict['lefteye'])
            neuroticismListLeftEye = self.modelDict['lefteye']['neuroticism'].predict(featuresDict['lefteye'])
            agreeablenessListLeftEye = self.modelDict['lefteye']['agreeableness'].predict(featuresDict['lefteye'])
            conscientiousnessListLeftEye = self.modelDict['lefteye']['conscientiousness'].predict(
                featuresDict['lefteye'])

        if featuresDict['smile']:
            opennessListSmile = self.modelDict['smile']['openness'].predict(featuresDict['smile'])
            extraversionListSmile = self.modelDict['smile']['extraversion'].predict(featuresDict['smile'])
            neuroticismListSmile = self.modelDict['smile']['neuroticism'].predict(featuresDict['smile'])
            agreeablenessListSmile = self.modelDict['smile']['agreeableness'].predict(featuresDict['smile'])
            conscientiousnessListSmile = self.modelDict['smile']['conscientiousness'].predict(featuresDict['smile'])

        openness = defaultdict(float)
        extraversion = defaultdict(float)
        neuroticism = defaultdict(float)
        conscientiousness = defaultdict(float)
        agreeableness = defaultdict(float)

        openness["average"] = -1
        extraversion["average"] = -1
        neuroticism["average"] = -1
        conscientiousness["average"] = -1
        agreeableness["average"] = -1

        opennessList = None
        agreeablenessList = None
        conscientiousnessList = None
        extraversionList = None
        neuroticismList = None

        if opennessListFace or opennessListRightEye or opennessListLeftEye or opennessListSmile:
            opennessList = np.concatenate(
                (opennessListFace, opennessListLeftEye, opennessListRightEye, opennessListSmile))
            openness["average"] = np.average(opennessList)
            openness["min"] = np.min(opennessList)
            openness["max"] = np.max(opennessList)

        if extraversionListFace or extraversionListLeftEye or extraversionListRightEye or extraversionListSmile:
            extraversionList = np.concatenate(
                (extraversionListFace, extraversionListLeftEye, extraversionListRightEye, extraversionListSmile))
            extraversion["average"] = np.average(extraversionList)
            extraversion["min"] = np.min(extraversionList)
            extraversion["max"] = np.max(extraversionList)

        if neuroticismListFace or neuroticismListLeftEye or neuroticismListRightEye or neuroticismListSmile:
            neuroticismList = np.concatenate(
                (neuroticismListFace, neuroticismListLeftEye, neuroticismListRightEye, neuroticismListSmile))
            neuroticism["average"] = np.average(neuroticismList)
            neuroticism["min"] = np.min(neuroticismList)
            neuroticism["max"] = np.max(neuroticismList)

        if agreeablenessListFace or agreeablenessListLeftEye or agreeablenessListRightEye or agreeablenessListLeftEye or agreeablenessListSmile:
            agreeablenessList = np.concatenate(
                (agreeablenessListFace, agreeablenessListLeftEye, agreeablenessListRightEye, agreeablenessListSmile))
            agreeableness["average"] = np.average(agreeablenessList)
            agreeableness["min"] = np.min(agreeablenessList)
            agreeableness["max"] = np.max(agreeablenessList)

        if conscientiousnessListFace or conscientiousnessListLeftEye or conscientiousnessListRightEye or conscientiousnessListSmile:
            conscientiousnessList = np.concatenate((conscientiousnessListFace, conscientiousnessListLeftEye,
                                                    conscientiousnessListRightEye, conscientiousnessListSmile))
            conscientiousness["average"] = np.average(conscientiousnessList)
            conscientiousness["min"] = np.min(conscientiousnessList)
            conscientiousness["max"] = np.max(conscientiousnessList)

        return openness, extraversion, neuroticism, agreeableness, conscientiousness
