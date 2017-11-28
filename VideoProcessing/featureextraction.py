import cv2

from collections import defaultdict

from VideoProcessing.localbinarypatterns import LocalBinaryPatterns


class FeatureExtraction:
    def __init__(self, points, radius):
        self.points = points
        self.radius = radius

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