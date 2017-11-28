import pickle

import numpy as np
import xlsxwriter

from VideoProcessing.featureextraction import FeatureExtraction
from VideoProcessing.frameextraction import FrameExtraction


class ModeTester:
    def __init__(self, model):
        self.modelDict = model

    def test(self, videoCount = 1, videoInterval = 200):
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

        workbook = xlsxwriter.Workbook('rbfsvm_train.xlsx')

        worksheet = workbook.add_worksheet()

        bold = workbook.add_format({'bold': True})

        worksheet.write('A1', 'Openness (E)', bold)
        worksheet.write('B1', 'Openness (C)', bold)
        worksheet.write('C1', 'Error', bold)
        worksheet.write('D1', 'Extraversion (E)', bold)
        worksheet.write('E1', 'Extraversion (C)', bold)
        worksheet.write('F1', 'Error', bold)

        worksheet.write('G1', 'Neuroticism (E)', bold)
        worksheet.write('H1', 'Neuroticism (C)', bold)
        worksheet.write('I1', 'Error', bold)
        worksheet.write('J1', 'Agreeableness (E)', bold)
        worksheet.write('K1', 'Agreeableness (C)', bold)
        worksheet.write('L1', 'Error', bold)

        worksheet.write('M1', 'Conscientiousness (E)', bold)
        worksheet.write('N1', 'Conscientiousness (C)', bold)
        worksheet.write('O1', 'Error', bold)

        videosDataFile = open("AnnotationFiles/annotation_training.pkl", "rb")
        print('Loading data from pickle file.')
        videosData = pickle.load(videosDataFile, encoding='latin1')

        print('Getting names of all the video files.')
        videoNames = list(videosData['extraversion'].keys())

        videosFilePath = 'TrainingVideos/'

        frameExtractor = FrameExtraction(1, videoInterval, videoNames, videosFilePath)

        for i, videoName in enumerate(videoNames):

            print("Video number: {}".format(i + 1))

            facesData, smileData, leftEyeData, rightEyeData = frameExtractor.extract_single(videoName)

            featureExtractor = FeatureExtraction(24, 8)

            featuresDict = featureExtractor.extract(facesData, smileData, leftEyeData, rightEyeData)

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

            tempList = np.concatenate((opennessListFace, opennessListLeftEye, opennessListRightEye, opennessListSmile))

            worksheet.write(i + 1, 0, videosData['openness'][videoName])
            worksheet.write(i + 1, 1, sum(tempList) / len(tempList))
            worksheet.write(i + 1, 2, abs(videosData['openness'][videoName] -  sum(tempList) / len(tempList)))

            tempList = np.concatenate(
                (extraversionListFace, extraversionListLeftEye, extraversionListRightEye, extraversionListSmile))

            worksheet.write(i + 1, 3, videosData['extraversion'][videoName])
            worksheet.write(i + 1, 4, sum(tempList) / len(tempList))
            worksheet.write(i + 1, 5, abs(videosData['extraversion'][videoName] -  sum(tempList) / len(tempList)))

            tempList = np.concatenate(
                (neuroticismListFace, neuroticismListLeftEye, neuroticismListRightEye, neuroticismListSmile))

            worksheet.write(i + 1, 6, videosData['neuroticism'][videoName])
            worksheet.write(i + 1, 7, sum(tempList) / len(tempList))
            worksheet.write(i + 1, 8, abs(videosData['neuroticism'][videoName] -  sum(tempList) / len(tempList)))

            tempList = np.concatenate(
                (agreeablenessListFace, agreeablenessListLeftEye, agreeablenessListRightEye, agreeablenessListSmile))

            worksheet.write(i + 1, 9, videosData['agreeableness'][videoName])
            worksheet.write(i + 1, 10, sum(tempList) / len(tempList))
            worksheet.write(i + 1, 11, abs(videosData['agreeableness'][videoName] -  sum(tempList) / len(tempList)))

            tempList = np.concatenate((conscientiousnessListFace, conscientiousnessListLeftEye,
                                       conscientiousnessListRightEye, conscientiousnessListSmile))

            worksheet.write(i + 1, 12, videosData['conscientiousness'][videoName])
            worksheet.write(i + 1, 13, sum(tempList) / len(tempList))
            worksheet.write(i + 1, 14, abs(videosData['conscientiousness'][videoName] -  sum(tempList) / len(tempList)))

            if i is videoCount - 1:
                break
        workbook.close()