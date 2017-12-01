import pickle

import cv2
from sklearn.svm import LinearSVR

from VideoProcessing.localbinarypatterns import LocalBinaryPatterns
from VideoProcessing.frameextraction import FrameExtraction
from VideoProcessing.featureextraction import FeatureExtraction
from ModelGeneration.linearsvmmodel import LinearSVMModel
from ModelGeneration.kernelridgeregression import KernelRidgeModel
from ModelGeneration.svmrbfmodel import RBFSVMModel
from ModelGeneration.neuralnetworkmodel import NeuralNetworkModel
from ModelGeneration.modeltester import ModeTester

def test_accuracy(model, lData, rData, videoName, videosInfo):
    lframe = lData[0]

    rframe = rData[0]

    LBPTestConvertor = LocalBinaryPatterns(24, 8)

    lgrayData = cv2.cvtColor(lframe, cv2.COLOR_BGR2GRAY)
    rgrayData = cv2.cvtColor(rframe, cv2.COLOR_BGR2GRAY)
    hist1 = LBPTestConvertor.describe(lgrayData)
    hist2 = LBPTestConvertor.describe(rgrayData)

    hist3 = hist1 + hist2
    eps = 1e-7
    hist3 /= (hist3.sum() + eps)
    print(len(hist3))
    print(hist3)

    tempData = []
    tempData.append(hist3)

    print(model.predict(tempData))

    print(videosInfo['openness'][videoName])

def temp_loader(facesData, videoLabels):
    # loading the haarcascade XML files to detect facial features
    # face_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_frontalface_default.xml')
    left_eye_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_righteye_2splits.xml')
    smile_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_smile.xml')

    markedfacesData = []
    leftEyeData = []
    smileData = []
    rightEyeData = []
    # videoLabels = []
    smileVideoLabels = []
    rightEyeVideoLabels = []
    leftEyeVideoLabels = []
    loopbreak = False

    for face, videoName in zip(facesData, videoLabels):
        grayFace = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # # detecting both left and right
        leftEye = left_eye_cascade.detectMultiScale(grayFace)  # Detecting left eye
        rightEye = right_eye_cascade.detectMultiScale(grayFace)  # Detecting right eye
        smile = smile_cascade.detectMultiScale(grayFace)

        for (ex, ey, ew, eh) in leftEye:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
            leftEyeFrame = face[ey:ey + eh, ex:ex + ew]
            leftEyeData.append(leftEyeFrame)
            leftEyeVideoLabels.append(videoName)
            break
        for (ex, ey, ew, eh) in rightEye:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            rightEyeFrame = face[ey:ey + eh, ex:ex + ew]
            rightEyeData.append(rightEyeFrame)
            rightEyeVideoLabels.append(videoName)
            break
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(face, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            smileFrame = face[sy:sy + sh, sx:sx + sw]
            smileData.append(smileFrame)
            smileVideoLabels.append(videoName)
            break

        markedfacesData.append(face)
        # videoLabels.append(videoName)

    # return facesData, leftEyeData, rightEyeData, smileData, videoLabels, smileVideoLabels
    return markedfacesData, leftEyeData, rightEyeData, smileData, leftEyeVideoLabels, rightEyeVideoLabels, smileVideoLabels


# Defining Variables

# folder where the training videos are kept
videosFilePath = 'TrainingVideos/'

# loading the data of the training files
videosDataFile = open("AnnotationFiles/annotation_training.pkl", "rb")
print('Loading data from pickle file.')
videosData = pickle.load(videosDataFile, encoding='latin1')

print('Getting names of all the video files.')
videoNames = list(videosData['extraversion'].keys())

# frameExtractor = FrameExtraction(1000, 200, videoNames, videosFilePath)

print("Starting frame extraction")
# facesData, leftEyeData, rightEyeData, smileData, videoLabels, smileVideoLabels = frameExtractor.extract()


# facesPickle = open("ModelStorage/faces(1000).pickle", "rb")
# videoLabelPickle = open("ModelStorage/videolabel.pickle", "rb")
# markedfacesPickle = open("ModelStorage/markedfaces(1000).pickle", "rb")
# lefteyePickle = open("ModelStorage/lefteye.pickle", "rb")
# righteyePickle = open("ModelStorage/righteye.pickle", "rb")
# smilePickle = open("ModelStorage/smile.pickle", "rb")
# smileVideoLabelPickle = open("ModelStorage/smilevideolabel.pickle", "rb")
# leftEyeVideoLabelPickle = open("ModelStorage/lefteyevideolabel.pickle", "rb")
# rightEyeVideoLabelPickle = open("ModelStorage/righteyevideolabel.pickle", "rb")

# featuresDictPickle = open('ModelStorage/features.pickle', "rb")
# lablesDictPickle = open("ModelStorage/facelabels(1000).pickle", "rb")
#
#SVMModelDictPickle = open("ModelStorage/svmmodel.pickle", "rb")
#
# KernelRidgeDictPickle = open("ModelStorage/kernelridge.pickle", "wb")

facesPickle = open("ModelStorage/faces2.pickle", "wb")
facesLabelsPickle = open("ModelStorage/facelabels2.pickle", "wb")

#
# rbfSVMPickle = open("ModelStorage/rbfsvm.pickle", "rb")

# nnadamPickle = open("ModelStorage/nnadam.pickle", "rb")

print("loading data")

# facesData = pickle.load(facesPickle)
# videoLabels = pickle.load(videoLabelPickle)
# smileData = pickle.load(smilePickle)
# rightEyeData = pickle.load(righteyePickle)
# leftEyeData = pickle.load(lefteyePickle)
# smileLabels = pickle.load(smileVideoLabelPickle)
# rightEyeLabels = pickle.load(rightEyeVideoLabelPickle)
# leftEyeLabels = pickle.load(leftEyeVideoLabelPickle)

# featuresDict = pickle.load(featuresDictPickle)
# labelsDict = pickle.load(lablesDictPickle)

frameExtractor = FrameExtraction(1000, 200, videoNames, videosFilePath)

faces, facelables = frameExtractor.extract(1000)

print("done loading data")

pickle.dump(faces, facesPickle)
pickle.dump(facelables, facesLabelsPickle)

#featureExtractor = FeatureExtraction(24, 8)

# print("Extracting features")
# featuresDict = featureExtractor.extract(facesData, smileData, leftEyeData, rightEyeData)
#
# print("Extracting labels")
# labelsDict = featureExtractor.make_feature_matrix(videosData, videoLabels, smileLabels, leftEyeLabels, rightEyeLabels)

# print(len(featuresDict))
# print(len(labelsDict))
#
# print(len(featuresDict['face']))
# print(len(labelsDict['face']))


# print("Starting to train model")
# SVMModel = LinearSVMModel(featuresDict, labelsDict)
#
# SVMModelDict = SVMModel.generate()

# SVMModelDict = pickle.load(SVMModelDictPickle)
#
# SVMModel = LinearSVMModel()
#
# SVMModel.test(SVMModelDict, 100)

#
# KernelModel = KernelRidgeModel(featuresDict, labelsDict)
#
# KernelRidgeModelDict = KernelModel.generate()

# rbfmodel = RBFSVMModel()
#
# rbfModelDict = pickle.load(rbfSVMPickle)
#
# print(rbfModelDict)
#
# rbfmodel.test(rbfModelDict, 100)


# ModelDict = pickle.load(rbfSVMPickle)
#
# Modeltester = ModeTester(ModelDict)
#
# Modeltester.test(200)

# nnModelDict = nnModel.generate()
#
# pickle.dump(nnModelDict, nnadamPickle)

#pickle.dump(rbfModelDict, rbfSVMPickle)

#pickle.dump(KernelRidgeModelDict, KernelRidgeDictPickle)

# pickle.dump(featuresDict, featuresDictPickle)
# pickle.dump(labelsDict, lablesDictPickle)

# markedfacesData, leftEyeData, rightEyeData, smileData, leftEyeVideoLabels, rightEyeVideoLabels, smileVideoLabels = temp_loader(
#     facesData, videoLabels)
# leftEyeData = pickle.load(lefteyePickle)
# rightEyeData = pickle.load(righteyePickle)
# smileData = pickle.load(smilePickle)
#
# smileVideoLabels = pickle.load(smileVideoLabelPickle)


# print("printing data")
# print(videoLabels)
# print(smileVideoLabels)
#
# for face in facesData:
#     cv2.imshow("Faces", face)
#     cv2.waitKey(0)
# pickle.dump(markedfacesData, markedfacesPickle)
# pickle.dump(leftEyeData, lefteyePickle)
# pickle.dump(rightEyeData, righteyePickle)
# pickle.dump(smileData, smilePickle)
# pickle.dump(smileVideoLabels, smileVideoLabelPickle)
# pickle.dump(leftEyeVideoLabels, leftEyeVideoLabelPickle)
# pickle.dump(rightEyeVideoLabels, rightEyeVideoLabelPickle)

# LBPConvertor = LocalBinaryPatterns(24, 8)
#
# opennessLabels = []
# extraversionLabels = []
# neuroticismLabels = []
# agreeablenessLabels = []
# conscientiousnessLabels = []
# #
# eyesValues = []
# faceValues = []
# smileValues = []
#
# # print("Starting to print faces")
# # print(len(facesData))
# for r, l, x, y, z in zip(rightEyeData, leftEyeData, facesData, smileData, videoLabels):
#     extraversionLabels.append(videosData['extraversion'][z])
#     neuroticismLabels.append(videosData['neuroticism'][z])
#     conscientiousnessLabels.append(videosData['conscientiousness'][z])
#     opennessLabels.append(videosData['openness'][z])
#     agreeablenessLabels.append(videosData['agreeableness'][z])
#
#     # convert both left and right images to greyscale
#     grayRight = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
#     grayLeft = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
#     grayFace = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
#     graySmile = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
#
#     # generate Local binary patterns for both left and right eye images
#     histRight = LBPConvertor.describe(grayRight)
#     histLeft = LBPConvertor.describe(grayLeft)
#     histFace = LBPConvertor.describe(grayFace)
#     histSmile = LBPConvertor.describe(graySmile)
#
#     # normalize the value of histogram
#     histBoth = histLeft + histRight
#     eps = 1e-7
#     histBoth /= (histBoth.sum() + eps)
#
#     eyesValues.append(histBoth)
#     faceValues.append(histFace)
#     smileValues.append(histSmile)
#
# neuroticismModelEyes = LinearSVR(random_state=0)
# neuroticismModelFace = LinearSVR(random_state=0)
# neuroticismModelSmile = LinearSVR(random_state=0)
#
# extraversionModelEyes = LinearSVR(random_state=0)
# extraversionModelFace = LinearSVR(random_state=0)
# extraversionModelSmile = LinearSVR(random_state=0)
#
# conscientiousnessModelEyes = LinearSVR(random_state=0)
# conscientiousnessModelFace = LinearSVR(random_state=0)
# conscientiousnessModelSmile = LinearSVR(random_state=0)
#
# agreeablenessModelEyes = LinearSVR(random_state=0)
# agreeablenessModelFace = LinearSVR(random_state=0)
# agreeablenessModelSmile = LinearSVR(random_state=0)
#
# opennessModelEyes = LinearSVR(random_state=0)
# opennessModelFace = LinearSVR(random_state=0)
# opennessModelSmile = LinearSVR(random_state=0)
#
# neuroticismModelEyes.fit(eyesValues, neuroticismLabels)
# neuroticismModelFace.fit(faceValues, neuroticismLabels)
# neuroticismModelSmile.fit(smileValues, neuroticismLabels)
#
# agreeablenessModelEyes.fit(eyesValues, agreeablenessLabels)
# agreeablenessModelFace.fit(faceValues, agreeablenessLabels)
# agreeablenessModelSmile.fit(smileValues, agreeablenessLabels)
#
# opennessModelEyes.fit(eyesValues, opennessLabels)
# opennessModelFace.fit(faceValues, opennessLabels)
# opennessModelSmile.fit(smileValues, opennessLabels)
#
# extraversionModelEyes.fit(eyesValues, extraversionLabels)
# extraversionModelFace.fit(faceValues, extraversionLabels)
# extraversionModelSmile.fit(smileValues, extraversionLabels)
#
# conscientiousnessModelEyes.fit(eyesValues, conscientiousnessLabels)
# conscientiousnessModelFace.fit(faceValues, conscientiousnessLabels)
# conscientiousnessModelSmile.fit(smileValues, conscientiousnessLabels)
#
# test_accuracy(opennessModelFace, rightEyeData, leftEyeData, videoLabels[0], videosData)
#
#
# #
# # testImage = rightEyeData[0]
# #
# # gray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
# #
# # hist1 = LBPConvertor.describe(gray)
# #
# # prediction = []
# # prediction.append(hist1)
# # print(neuroticismModel.predict(prediction))
# # print(videosData['neuroticism'][videoNames[0]])
# #
# # cv2.destroyAllWindows()
