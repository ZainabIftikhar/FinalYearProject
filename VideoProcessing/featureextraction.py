import pickle

import cv2

from collections import defaultdict

from VideoProcessing.localbinarypatterns import LocalBinaryPatterns


class FeatureExtraction:
    def __init__(self, points, radius):
        self.points = points
        self.radius = radius



    def extract_all(self):

        LBPGenerator = LocalBinaryPatterns(self.points, self.radius)

        faceTrainFileNames = ['faces1.pickle','faces2.pickle', 'faces3.pickle', 'faces4.pickle', 'faces5.pickle', 'faces6.pickle']

        faceTestFileNames =  ['faces_test1.pickle', 'faces_test2.pickle']

        faceValFileNames = ['faces_val1.pickle', 'faces_val2.pickle']

        faceTrainLabelFileNames = ['facelabels1.pickle','facelabels2.pickle', 'facelabels3.pickle', 'facelabels4.pickle', 'facelabels5.pickle',
                              'facelabels6.pickle']
        faceTestLabelFileNames = ['facelabels_test1.pickle', 'facelabels_test2.pickle']

        faceValLabelFileNames = ['facelabels_val1.pickle', 'facelabels_val2.pickle']

        TestVideosData = pickle.load(open("AnnotationFiles/annotation_test.pkl", "rb"), encoding='latin1')
        TrainVideosData = pickle.load(open("AnnotationFiles/annotation_training.pkl", "rb"), encoding='latin1')
        ValVideosData = pickle.load(open("AnnotationFiles/annotation_validation.pkl", "rb"), encoding='latin1')

        facesTestHist = []
        smileTestHist = []
        rightEyeTestHist = []
        leftEyeTestHist = []


        facesTrainHist = []
        smileTrainHist = []
        rightEyeTrainHist = []
        leftEyeTrainHist = []


        facesValHist = []
        smileValHist = []
        rightEyeValHist = []
        leftEyeValHist = []

        agreeablenessFaceTrainLabels = []
        opennessFaceTrainLabels = []
        neuroticismFaceTrainLabels = []
        extraversionFaceTrainLabels = []
        conscientiousnessFaceTrainLabels = []

        agreeablenessLeftEyeTrainLabels = []
        opennessLeftEyeTrainLabels = []
        neuroticismLeftEyeTrainLabels = []
        extraversionLeftEyeTrainLabels = []
        conscientiousnessLeftEyeTrainLabels = []

        agreeablenessRightEyeTrainLabels = []
        opennessRightEyeTrainLabels = []
        neuroticismRightEyeTrainLabels = []
        extraversionRightEyeTrainLabels = []
        conscientiousnessRightEyeTrainLabels = []

        agreeablenessSmileTrainLabels = []
        opennessSmileTrainLabels = []
        neuroticismSmileTrainLabels = []
        extraversionSmileTrainLabels = []
        conscientiousnessSmileTrainLabels = []


        agreeablenessFaceTestLabels = []
        opennessFaceTestLabels = []
        neuroticismFaceTestLabels = []
        extraversionFaceTestLabels = []
        conscientiousnessFaceTestLabels = []

        agreeablenessLeftEyeTestLabels = []
        opennessLeftEyeTestLabels = []
        neuroticismLeftEyeTestLabels = []
        extraversionLeftEyeTestLabels = []
        conscientiousnessLeftEyeTestLabels = []

        agreeablenessRightEyeTestLabels = []
        opennessRightEyeTestLabels = []
        neuroticismRightEyeTestLabels = []
        extraversionRightEyeTestLabels = []
        conscientiousnessRightEyeTestLabels = []

        agreeablenessSmileTestLabels = []
        opennessSmileTestLabels = []
        neuroticismSmileTestLabels = []
        extraversionSmileTestLabels = []
        conscientiousnessSmileTestLabels = []


        agreeablenessFaceValLabels = []
        opennessFaceValLabels = []
        neuroticismFaceValLabels = []
        extraversionFaceValLabels = []
        conscientiousnessFaceValLabels = []

        agreeablenessLeftEyeValLabels = []
        opennessLeftEyeValLabels = []
        neuroticismLeftEyeValLabels = []
        extraversionLeftEyeValLabels = []
        conscientiousnessLeftEyeValLabels = []

        agreeablenessRightEyeValLabels = []
        opennessRightEyeValLabels = []
        neuroticismRightEyeValLabels = []
        extraversionRightEyeValLabels = []
        conscientiousnessRightEyeValLabels = []

        agreeablenessSmileValLabels = []
        opennessSmileValLabels = []
        neuroticismSmileValLabels = []
        extraversionSmileValLabels = []
        conscientiousnessSmileValLabels = []

        folderLocation = 'ModelStorage/'
        lefteye = 'lefteye_'
        righteye = 'righteye_'
        smile = 'smile_'
        label = 'label_'

        for facefile, facelabelfile in zip(faceTrainFileNames, faceTrainLabelFileNames):
            facePickle = open(folderLocation + facefile, "rb")
            print("Face file {} openend".format(facefile))

            facelabelPickle = open(folderLocation + facelabelfile, 'rb')
            print("Face label file {} opened".format(facelabelfile))


            lefteyePickle = open(folderLocation + lefteye + facefile, "rb")
            print("Left eye file {} openend".format(folderLocation + lefteye + facefile))

            lefteyelabelPickle = open(folderLocation + lefteye + label + facefile, "rb")
            print("Left eye label file {} openend".format(folderLocation + lefteye + label + facefile))


            righteyePickle = open(folderLocation + righteye + facefile, "rb")
            print("Right eye file {} openend".format(folderLocation + righteye + facefile))

            righteyelabelPickle = open(folderLocation + righteye + label + facefile, "rb")
            print("Right eye label file {} openend".format(folderLocation + righteye + label + facefile))


            smilePickle = open(folderLocation + smile + facefile, "rb")
            print("Smile file {} openend".format(folderLocation + smile + facefile))

            smilelabelPickle = open(folderLocation + smile + label + facefile, "rb")
            print("Smile label file {} openend".format(folderLocation + smile + label + facefile))

            facesData = pickle.load(facePickle)
            faceLabels = pickle.load(facelabelPickle)

            smileData = pickle.load(smilePickle)
            smileLabels = pickle.load(smilelabelPickle)


            leftEyeData = pickle.load(lefteyePickle)
            leftEyeLabels = pickle.load(lefteyelabelPickle)

            rightEyeData = pickle.load(righteyePickle)
            rightEyeLabels = pickle.load(righteyelabelPickle)

            for i, (face, faceLabel) in enumerate(zip(facesData, faceLabels)):
                print("Feature extraction of face number: {}".format(i))
                grayface = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                facesTrainHist.append(LBPGenerator.describe(grayface))

                extraversionFaceTrainLabels.append(TrainVideosData['extraversion'][faceLabel])
                neuroticismFaceTrainLabels.append(TrainVideosData['neuroticism'][faceLabel])
                conscientiousnessFaceTrainLabels.append(TrainVideosData['conscientiousness'][faceLabel])
                opennessFaceTrainLabels.append(TrainVideosData['openness'][faceLabel])
                agreeablenessFaceTrainLabels.append(TrainVideosData['agreeableness'][faceLabel])

            for i, (smile, smileLabel) in enumerate(zip(smileData, smileLabels)):
                print("Feature extraction of smile number: {}".format(i))
                graySmile = cv2.cvtColor(smile, cv2.COLOR_BGR2GRAY)
                smileTrainHist.append(LBPGenerator.describe(graySmile))

                extraversionSmileTrainLabels.append(TrainVideosData['extraversion'][smileLabel])
                neuroticismSmileTrainLabels.append(TrainVideosData['neuroticism'][smileLabel])
                conscientiousnessSmileTrainLabels.append(TrainVideosData['conscientiousness'][smileLabel])
                opennessSmileTrainLabels.append(TrainVideosData['openness'][smileLabel])
                agreeablenessSmileTrainLabels.append(TrainVideosData['agreeableness'][smileLabel])

            for i, (leftEye, leftEyeLabel) in enumerate(zip(leftEyeData, leftEyeLabels)):
                print("Feature extraction of lefteye number: {}".format(i))
                grayLeftEye = cv2.cvtColor(leftEye, cv2.COLOR_BGR2GRAY)
                leftEyeTrainHist.append(LBPGenerator.describe(grayLeftEye))

                extraversionLeftEyeTrainLabels.append(TrainVideosData['extraversion'][leftEyeLabel])
                neuroticismLeftEyeTrainLabels.append(TrainVideosData['neuroticism'][leftEyeLabel])
                conscientiousnessLeftEyeTrainLabels.append(TrainVideosData['conscientiousness'][leftEyeLabel])
                opennessLeftEyeTrainLabels.append(TrainVideosData['openness'][leftEyeLabel])
                agreeablenessLeftEyeTrainLabels.append(TrainVideosData['agreeableness'][leftEyeLabel])

            for i, (rightEye, rightEyeLabel) in enumerate(zip(rightEyeData, rightEyeLabels)):
                print("Feature extraction of righteye number: {}".format(i))
                grayRightEye = cv2.cvtColor(righteye, cv2.COLOR_BGR2GRAY)
                rightEyeTrainHist.append(LBPGenerator.describe(grayRightEye))

                extraversionRightEyeTrainLabels.append(TrainVideosData['extraversion'][rightEyeLabel])
                neuroticismRightEyeTrainLabels.append(TrainVideosData['neuroticism'][rightEyeLabel])
                conscientiousnessRightEyeTrainLabels.append(TrainVideosData['conscientiousness'][rightEyeLabel])
                opennessRightEyeTrainLabels.append(TrainVideosData['openness'][rightEyeLabel])
                agreeablenessRightEyeTrainLabels.append(TrainVideosData['agreeableness'][rightEyeLabel])


        for facefile, facelabelfile in zip(faceTestFileNames, faceTestLabelFileNames):
            facePickle = open(folderLocation + facefile, "rb")
            print("Face file {} openend".format(facefile))

            facelabelPickle = open(folderLocation + facelabelfile, 'rb')
            print("Face label file {} opened".format(facelabelfile))

            lefteyePickle = open(folderLocation + lefteye + facefile, "rb")
            print("Left eye file {} openend".format(folderLocation + lefteye + facefile))

            lefteyelabelPickle = open(folderLocation + lefteye + label + facefile, "rb")
            print("Left eye label file {} openend".format(folderLocation + lefteye + label + facefile))

            righteyePickle = open(folderLocation + righteye + facefile, "rb")
            print("Right eye file {} openend".format(folderLocation + righteye + facefile))

            righteyelabelPickle = open(folderLocation + righteye + label + facefile, "rb")
            print("Right eye label file {} openend".format(folderLocation + righteye + label + facefile))

            smilePickle = open(folderLocation + smile + facefile, "rb")
            print("Smile file {} openend".format(folderLocation + smile + facefile))

            smilelabelPickle = open(folderLocation + smile + label + facefile, "rb")
            print("Smile label file {} openend".format(folderLocation + smile + label + facefile))

            facesData = pickle.load(facePickle)
            faceLabels = pickle.load(facelabelPickle)

            smileData = pickle.load(smilePickle)
            smileLabels = pickle.load(smilelabelPickle)

            leftEyeData = pickle.load(lefteyePickle)
            leftEyeLabels = pickle.load(lefteyelabelPickle)

            rightEyeData = pickle.load(righteyePickle)
            rightEyeLabels = pickle.load(righteyelabelPickle)

            for i, (face, faceLabel) in enumerate(zip(facesData, faceLabels)):
                print("Feature extraction of face number: {}".format(i))
                grayface = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                facesTestHist.append(LBPGenerator.describe(grayface))

                extraversionFaceTestLabels.append(TestVideosData['extraversion'][faceLabel])
                neuroticismFaceTestLabels.append(TestVideosData['neuroticism'][faceLabel])
                conscientiousnessFaceTestLabels.append(TestVideosData['conscientiousness'][faceLabel])
                opennessFaceTestLabels.append(TestVideosData['openness'][faceLabel])
                agreeablenessFaceTestLabels.append(TestVideosData['agreeableness'][faceLabel])

            for i, (smile, smileLabel) in enumerate(zip(smileData, smileLabels)):
                print("Feature extraction of smile number: {}".format(i))
                graySmile = cv2.cvtColor(smile, cv2.COLOR_BGR2GRAY)
                smileTestHist.append(LBPGenerator.describe(graySmile))

                extraversionSmileTestLabels.append(TestVideosData['extraversion'][smileLabel])
                neuroticismSmileTestLabels.append(TestVideosData['neuroticism'][smileLabel])
                conscientiousnessSmileTestLabels.append(TestVideosData['conscientiousness'][smileLabel])
                opennessSmileTestLabels.append(TestVideosData['openness'][smileLabel])
                agreeablenessSmileTestLabels.append(TestVideosData['agreeableness'][smileLabel])

            for i, (leftEye, leftEyeLabel) in enumerate(zip(leftEyeData, leftEyeLabels)):
                print("Feature extraction of lefteye number: {}".format(i))
                grayLeftEye = cv2.cvtColor(leftEye, cv2.COLOR_BGR2GRAY)
                leftEyeTestHist.append(LBPGenerator.describe(grayLeftEye))

                extraversionLeftEyeTestLabels.append(TestVideosData['extraversion'][leftEyeLabel])
                neuroticismLeftEyeTestLabels.append(TestVideosData['neuroticism'][leftEyeLabel])
                conscientiousnessLeftEyeTestLabels.append(TestVideosData['conscientiousness'][leftEyeLabel])
                opennessLeftEyeTestLabels.append(TestVideosData['openness'][leftEyeLabel])
                agreeablenessLeftEyeTestLabels.append(TestVideosData['agreeableness'][leftEyeLabel])

            for i, (rightEye, rightEyeLabel) in enumerate(zip(rightEyeData, rightEyeLabels)):
                print("Feature extraction of righteye number: {}".format(i))
                grayRightEye = cv2.cvtColor(righteye, cv2.COLOR_BGR2GRAY)
                rightEyeTestHist.append(LBPGenerator.describe(grayRightEye))

                extraversionRightEyeTestLabels.append(TestVideosData['extraversion'][rightEyeLabel])
                neuroticismRightEyeTestLabels.append(TestVideosData['neuroticism'][rightEyeLabel])
                conscientiousnessRightEyeTestLabels.append(TestVideosData['conscientiousness'][rightEyeLabel])
                opennessRightEyeTestLabels.append(TestVideosData['openness'][rightEyeLabel])
                agreeablenessRightEyeTestLabels.append(TestVideosData['agreeableness'][rightEyeLabel])

        for facefile, facelabelfile in zip(faceValFileNames, faceValLabelFileNames):
            facePickle = open(folderLocation + facefile, "rb")
            print("Face file {} openend".format(facefile))

            facelabelPickle = open(folderLocation + facelabelfile, 'rb')
            print("Face label file {} opened".format(facelabelfile))

            lefteyePickle = open(folderLocation + lefteye + facefile, "rb")
            print("Left eye file {} openend".format(folderLocation + lefteye + facefile))

            lefteyelabelPickle = open(folderLocation + lefteye + label + facefile, "rb")
            print("Left eye label file {} openend".format(folderLocation + lefteye + label + facefile))

            righteyePickle = open(folderLocation + righteye + facefile, "rb")
            print("Right eye file {} openend".format(folderLocation + righteye + facefile))

            righteyelabelPickle = open(folderLocation + righteye + label + facefile, "rb")
            print("Right eye label file {} openend".format(folderLocation + righteye + label + facefile))

            smilePickle = open(folderLocation + smile + facefile, "rb")
            print("Smile file {} openend".format(folderLocation + smile + facefile))

            smilelabelPickle = open(folderLocation + smile + label + facefile, "rb")
            print("Smile label file {} openend".format(folderLocation + smile + label + facefile))

            facesData = pickle.load(facePickle)
            faceLabels = pickle.load(facelabelPickle)

            smileData = pickle.load(smilePickle)
            smileLabels = pickle.load(smilelabelPickle)

            leftEyeData = pickle.load(lefteyePickle)
            leftEyeLabels = pickle.load(lefteyelabelPickle)

            rightEyeData = pickle.load(righteyePickle)
            rightEyeLabels = pickle.load(righteyelabelPickle)

            for i, (face, faceLabel) in enumerate(zip(facesData, faceLabels)):
                print("Feature extraction of face number: {}".format(i))
                grayface = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                facesValHist.append(LBPGenerator.describe(grayface))

                extraversionFaceValLabels.append(ValVideosData['extraversion'][faceLabel])
                neuroticismFaceValLabels.append(ValVideosData['neuroticism'][faceLabel])
                conscientiousnessFaceValLabels.append(ValVideosData['conscientiousness'][faceLabel])
                opennessFaceValLabels.append(ValVideosData['openness'][faceLabel])
                agreeablenessFaceValLabels.append(ValVideosData['agreeableness'][faceLabel])

            for i, (smile, smileLabel) in enumerate(zip(smileData, smileLabels)):
                print("Feature extraction of smile number: {}".format(i))
                graySmile = cv2.cvtColor(smile, cv2.COLOR_BGR2GRAY)
                smileValHist.append(LBPGenerator.describe(graySmile))

                extraversionSmileValLabels.append(ValVideosData['extraversion'][smileLabel])
                neuroticismSmileValLabels.append(ValVideosData['neuroticism'][smileLabel])
                conscientiousnessSmileValLabels.append(ValVideosData['conscientiousness'][smileLabel])
                opennessSmileValLabels.append(ValVideosData['openness'][smileLabel])
                agreeablenessSmileValLabels.append(ValVideosData['agreeableness'][smileLabel])

            for i, (leftEye, leftEyeLabel) in enumerate(zip(leftEyeData, leftEyeLabels)):
                print("Feature extraction of lefteye number: {}".format(i))
                grayLeftEye = cv2.cvtColor(leftEye, cv2.COLOR_BGR2GRAY)
                leftEyeValHist.append(LBPGenerator.describe(grayLeftEye))

                extraversionLeftEyeValLabels.append(ValVideosData['extraversion'][leftEyeLabel])
                neuroticismLeftEyeValLabels.append(ValVideosData['neuroticism'][leftEyeLabel])
                conscientiousnessLeftEyeValLabels.append(ValVideosData['conscientiousness'][leftEyeLabel])
                opennessLeftEyeValLabels.append(ValVideosData['openness'][leftEyeLabel])
                agreeablenessLeftEyeValLabels.append(ValVideosData['agreeableness'][leftEyeLabel])

            for i, (rightEye, rightEyeLabel) in enumerate(zip(rightEyeData, rightEyeLabels)):
                print("Feature extraction of righteye number: {}".format(i))
                grayRightEye = cv2.cvtColor(righteye, cv2.COLOR_BGR2GRAY)
                rightEyeValHist.append(LBPGenerator.describe(grayRightEye))

                extraversionRightEyeValLabels.append(ValVideosData['extraversion'][rightEyeLabel])
                neuroticismRightEyeValLabels.append(ValVideosData['neuroticism'][rightEyeLabel])
                conscientiousnessRightEyeValLabels.append(ValVideosData['conscientiousness'][rightEyeLabel])
                opennessRightEyeValLabels.append(ValVideosData['openness'][rightEyeLabel])
                agreeablenessRightEyeValLabels.append(ValVideosData['agreeableness'][rightEyeLabel])

        featuresDictTrain = defaultdict(dict)

        featuresDictTrain['face']['hist'] = facesTrainHist
        featuresDictTrain['face']['openness'] = opennessFaceTrainLabels
        featuresDictTrain['face']['extraversion'] = extraversionFaceTrainLabels
        featuresDictTrain['face']['neuroticism'] = neuroticismFaceTrainLabels
        featuresDictTrain['face']['agreeableness'] = agreeablenessFaceTrainLabels
        featuresDictTrain['face']['conscientiousness'] = conscientiousnessFaceTrainLabels


        featuresDictTrain['lefteye']['hist'] = leftEyeTrainHist
        featuresDictTrain['lefteye']['openness'] = opennessLeftEyeTrainLabels
        featuresDictTrain['lefteye']['extraversion'] = extraversionLeftEyeTrainLabels
        featuresDictTrain['lefteye']['neuroticism'] = neuroticismLeftEyeTrainLabels
        featuresDictTrain['lefteye']['agreeableness'] = agreeablenessLeftEyeTrainLabels
        featuresDictTrain['lefteye']['conscientiousness'] = conscientiousnessLeftEyeTrainLabels


        featuresDictTrain['righteye']['hist'] = rightEyeTrainHist
        featuresDictTrain['righteye']['openness'] = opennessRightEyeTrainLabels
        featuresDictTrain['righteye']['extraversion'] = extraversionRightEyeTrainLabels
        featuresDictTrain['righteye']['neuroticism'] = neuroticismRightEyeTrainLabels
        featuresDictTrain['righteye']['agreeableness'] = agreeablenessRightEyeTrainLabels
        featuresDictTrain['righteye']['conscientiousness'] = conscientiousnessRightEyeTrainLabels


        featuresDictTrain['smile']['hist'] = smileTrainHist
        featuresDictTrain['smile']['openness'] = opennessSmileTrainLabels
        featuresDictTrain['smile']['extraversion'] = extraversionSmileTrainLabels
        featuresDictTrain['smile']['neuroticism'] = neuroticismSmileTrainLabels
        featuresDictTrain['smile']['agreeableness'] = agreeablenessSmileTrainLabels
        featuresDictTrain['smile']['conscientiousness'] = conscientiousnessSmileTrainLabels

        featuresDictTest = defaultdict(dict)

        featuresDictTest['face']['hist'] = facesTestHist
        featuresDictTest['face']['openness'] = opennessFaceTestLabels
        featuresDictTest['face']['extraversion'] = extraversionFaceTestLabels
        featuresDictTest['face']['neuroticism'] = neuroticismFaceTestLabels
        featuresDictTest['face']['agreeableness'] = agreeablenessFaceTestLabels
        featuresDictTest['face']['conscientiousness'] = conscientiousnessFaceTestLabels

        featuresDictTest['lefteye']['hist'] = leftEyeTestHist
        featuresDictTest['lefteye']['openness'] = opennessLeftEyeTestLabels
        featuresDictTest['lefteye']['extraversion'] = extraversionLeftEyeTestLabels
        featuresDictTest['lefteye']['neuroticism'] = neuroticismLeftEyeTestLabels
        featuresDictTest['lefteye']['agreeableness'] = agreeablenessLeftEyeTestLabels
        featuresDictTest['lefteye']['conscientiousness'] = conscientiousnessLeftEyeTestLabels

        featuresDictTest['righteye']['hist'] = rightEyeTestHist
        featuresDictTest['righteye']['openness'] = opennessRightEyeTestLabels
        featuresDictTest['righteye']['extraversion'] = extraversionRightEyeTestLabels
        featuresDictTest['righteye']['neuroticism'] = neuroticismRightEyeTestLabels
        featuresDictTest['righteye']['agreeableness'] = agreeablenessRightEyeTestLabels
        featuresDictTest['righteye']['conscientiousness'] = conscientiousnessRightEyeTestLabels

        featuresDictTest['smile']['hist'] = smileTestHist
        featuresDictTest['smile']['openness'] = opennessSmileTestLabels
        featuresDictTest['smile']['extraversion'] = extraversionSmileTestLabels
        featuresDictTest['smile']['neuroticism'] = neuroticismSmileTestLabels
        featuresDictTest['smile']['agreeableness'] = agreeablenessSmileTestLabels
        featuresDictTest['smile']['conscientiousness'] = conscientiousnessSmileTestLabels


        featuresDictVal = defaultdict(dict)

        featuresDictVal['face']['hist'] = facesValHist
        featuresDictVal['face']['openness'] = opennessFaceValLabels
        featuresDictVal['face']['extraversion'] = extraversionFaceValLabels
        featuresDictVal['face']['neuroticism'] = neuroticismFaceValLabels
        featuresDictVal['face']['agreeableness'] = agreeablenessFaceValLabels
        featuresDictVal['face']['conscientiousness'] = conscientiousnessFaceValLabels

        featuresDictVal['lefteye']['hist'] = leftEyeValHist
        featuresDictVal['lefteye']['openness'] = opennessLeftEyeValLabels
        featuresDictVal['lefteye']['extraversion'] = extraversionLeftEyeValLabels
        featuresDictVal['lefteye']['neuroticism'] = neuroticismLeftEyeValLabels
        featuresDictVal['lefteye']['agreeableness'] = agreeablenessLeftEyeValLabels
        featuresDictVal['lefteye']['conscientiousness'] = conscientiousnessLeftEyeValLabels

        featuresDictVal['righteye']['hist'] = rightEyeValHist
        featuresDictVal['righteye']['openness'] = opennessRightEyeValLabels
        featuresDictVal['righteye']['extraversion'] = extraversionRightEyeValLabels
        featuresDictVal['righteye']['neuroticism'] = neuroticismRightEyeValLabels
        featuresDictVal['righteye']['agreeableness'] = agreeablenessRightEyeValLabels
        featuresDictVal['righteye']['conscientiousness'] = conscientiousnessRightEyeValLabels

        featuresDictVal['smile']['hist'] = smileValHist
        featuresDictVal['smile']['openness'] = opennessSmileValLabels
        featuresDictVal['smile']['extraversion'] = extraversionSmileValLabels
        featuresDictVal['smile']['neuroticism'] = neuroticismSmileValLabels
        featuresDictVal['smile']['agreeableness'] = agreeablenessSmileValLabels
        featuresDictVal['smile']['conscientiousness'] = conscientiousnessSmileValLabels


        return featuresDictTrain, featuresDictTest, featuresDictVal

    def extract(self, facesData, smileData, leftEyeData, rightEyeData):

        facesHist = []
        smileHist = []
        rightEyeHist = []
        leftEyeHist = []

        LBPGenerator = LocalBinaryPatterns(self.points, self.radius)

        for i, face in enumerate(facesData):
            print("Feature extraction of face number: {}".format(i))
            grayface = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            facesHist.append(LBPGenerator.describe(grayface))

        for i, smile in enumerate(smileData):
            print("Feature extraction of smile number: {}".format(i))
            graySmile = cv2.cvtColor(smile, cv2.COLOR_BGR2GRAY)
            smileHist.append(LBPGenerator.describe(graySmile))

        for i, righteye in enumerate(rightEyeData):
            print("Feature extraction of righteye number: {}".format(i))
            grayRightEye = cv2.cvtColor(righteye, cv2.COLOR_BGR2GRAY)
            rightEyeHist.append(LBPGenerator.describe(grayRightEye))

        for i, leftEye in enumerate(leftEyeData):
            print("Feature extraction of lefteye number: {}".format(i))
            grayLeftEye = cv2.cvtColor(leftEye, cv2.COLOR_BGR2GRAY)
            leftEyeHist.append(LBPGenerator.describe(grayLeftEye))

        self.facesHist = facesHist
        self.smileHist = smileHist
        self.rightEyeHist = rightEyeHist
        self.leftEyeHist = leftEyeHist

        featuresDict = defaultdict(dict)

        featuresDict['face'] = facesHist
        featuresDict['smile'] = smileHist
        featuresDict['righteye'] = rightEyeHist
        featuresDict['lefteye'] = leftEyeHist

        return featuresDict

    def make_feature_matrix(self, videosData, facesLabels, smileLabels, leftEyeLabels, rightEyeLabels):

        agreeablenessFaceLabels = []
        opennessFaceLabels = []
        neuroticismFaceLabels = []
        extraversionFaceLabels = []
        conscientiousnessFaceLabels = []

        agreeablenessLeftEyeLabels = []
        opennessLeftEyeLabels = []
        neuroticismLeftEyeLabels = []
        extraversionLeftEyeLabels = []
        conscientiousnessLeftEyeFaceLabels = []

        agreeablenessRightEyeLabels = []
        opennessRightEyeLabels = []
        neuroticismRightEyeLabels = []
        extraversionRightEyeLabels = []
        conscientiousnessRightEyeLabels = []

        agreeablenessSmileLabels = []
        opennessSmileLabels = []
        neuroticismSmileLabels = []
        extraversionSmileLabels = []
        conscientiousnessSmileLabels = []

        labelsDict = defaultdict(dict)

        for z in facesLabels:
            extraversionFaceLabels.append(videosData['extraversion'][z])
            neuroticismFaceLabels.append(videosData['neuroticism'][z])
            conscientiousnessFaceLabels.append(videosData['conscientiousness'][z])
            opennessFaceLabels.append(videosData['openness'][z])
            agreeablenessFaceLabels.append(videosData['agreeableness'][z])

        labelsDict['face']['extraversion'] = extraversionFaceLabels
        labelsDict['face']['neuroticism'] = neuroticismFaceLabels
        labelsDict['face']['conscientiousness'] = conscientiousnessFaceLabels
        labelsDict['face']['openness'] = opennessFaceLabels
        labelsDict['face']['agreeableness'] = agreeablenessFaceLabels

        for z in smileLabels:
            extraversionSmileLabels.append(videosData['extraversion'][z])
            neuroticismSmileLabels.append(videosData['neuroticism'][z])
            conscientiousnessSmileLabels.append(videosData['conscientiousness'][z])
            opennessSmileLabels.append(videosData['openness'][z])
            agreeablenessSmileLabels.append(videosData['agreeableness'][z])

        labelsDict['smile']['extraversion'] = extraversionSmileLabels
        labelsDict['smile']['neuroticism'] = neuroticismSmileLabels
        labelsDict['smile']['conscientiousness'] = conscientiousnessSmileLabels
        labelsDict['smile']['openness'] = opennessSmileLabels
        labelsDict['smile']['agreeableness'] = agreeablenessSmileLabels

        for z in rightEyeLabels:
            extraversionRightEyeLabels.append(videosData['extraversion'][z])
            neuroticismRightEyeLabels.append(videosData['neuroticism'][z])
            conscientiousnessRightEyeLabels.append(videosData['conscientiousness'][z])
            opennessRightEyeLabels.append(videosData['openness'][z])
            agreeablenessRightEyeLabels.append(videosData['agreeableness'][z])

        labelsDict['righteye']['extraversion'] = extraversionRightEyeLabels
        labelsDict['righteye']['neuroticism'] = neuroticismRightEyeLabels
        labelsDict['righteye']['conscientiousness'] = conscientiousnessRightEyeLabels
        labelsDict['righteye']['openness'] = opennessRightEyeLabels
        labelsDict['righteye']['agreeableness'] = agreeablenessRightEyeLabels

        for z in leftEyeLabels:
            extraversionLeftEyeLabels.append(videosData['extraversion'][z])
            neuroticismLeftEyeLabels.append(videosData['neuroticism'][z])
            conscientiousnessLeftEyeFaceLabels.append(videosData['conscientiousness'][z])
            opennessLeftEyeLabels.append(videosData['openness'][z])
            agreeablenessLeftEyeLabels.append(videosData['agreeableness'][z])

        labelsDict['lefteye']['extraversion'] = extraversionLeftEyeLabels
        labelsDict['lefteye']['neuroticism'] = neuroticismLeftEyeLabels
        labelsDict['lefteye']['conscientiousness'] = conscientiousnessLeftEyeFaceLabels
        labelsDict['lefteye']['openness'] = opennessLeftEyeLabels
        labelsDict['lefteye']['agreeableness'] = agreeablenessLeftEyeLabels

        return labelsDict

    def extract_single_frame(self, face, smile, leftEye, rightEye):

        LBPGenerator = LocalBinaryPatterns(self.points, self.radius)

        faceHist = []
        smileHist = []
        rightEyeHist = []
        leftEyeHist = []
        if face is not None:
            grayface = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faceHist.append(LBPGenerator.describe(grayface))

        if smile is not None:
            graySmile = cv2.cvtColor(smile, cv2.COLOR_BGR2GRAY)
            smileHist.append(LBPGenerator.describe(graySmile))

        if rightEye is not None:
            grayRightEye = cv2.cvtColor(rightEye, cv2.COLOR_BGR2GRAY)
            rightEyeHist.append(LBPGenerator.describe(grayRightEye))

        if leftEye is not None:
            grayLeftEye = cv2.cvtColor(leftEye, cv2.COLOR_BGR2GRAY)
            leftEyeHist.append(LBPGenerator.describe(grayLeftEye))

        featuresDict = defaultdict(dict)

        featuresDict['face'] = faceHist
        featuresDict['smile'] = smileHist
        featuresDict['righteye'] = rightEyeHist
        featuresDict['lefteye'] = leftEyeHist

        return featuresDict