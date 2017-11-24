import pickle

import cv2
from sklearn.svm import LinearSVR

from VideoProcessing.localbinarypatterns import LocalBinaryPatterns
from VideoProcessing.frameextraction import FrameExtraction

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


# Defining Variables

# folder where the training videos are kept
videosFilePath = 'TrainingVideos/'

# loading the data of the training files
videosDataFile = open("AnnotationFiles/annotation_training.pkl", "rb")
print('Loading data from pickle file.')
videosData = pickle.load(videosDataFile, encoding='latin1')

print('Getting names of all the video files.')
videoNames = list(videosData['extraversion'].keys())


frameExtractor  = FrameExtraction(30, 200, videoNames, videosFilePath)

print("Starting frame extraction")
facesData, leftEyeData, rightEyeData, smileData, videoLabels = frameExtractor.extract()

LBPConvertor = LocalBinaryPatterns(24, 8)

opennessLabels = []
extraversionLabels = []
neuroticismLabels = []
agreeablenessLabels = []
conscientiousnessLabels = []
#
eyesValues = []
faceValues = []
smileValues = []

# print("Starting to print faces")
# print(len(facesData))
for r, l, x, y, z in zip(rightEyeData, leftEyeData, facesData, smileData, videoLabels):

     extraversionLabels.append(videosData['extraversion'][z])
     neuroticismLabels.append(videosData['neuroticism'][z])
     conscientiousnessLabels.append(videosData['conscientiousness'][z])
     opennessLabels.append(videosData['openness'][z])
     agreeablenessLabels.append(videosData['agreeableness'][z])

     #convert both left and right images to greyscale
     grayRight = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
     grayLeft = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
     grayFace = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
     graySmile = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)

     #generate Local binary patterns for both left and right eye images
     histRight = LBPConvertor.describe(grayRight)
     histLeft = LBPConvertor.describe(grayLeft)
     histFace = LBPConvertor.describe(grayFace)
     histSmile = LBPConvertor.describe(graySmile)

     #normalize the value of histogram
     histBoth = histLeft + histRight
     eps = 1e-7
     histBoth /= (histBoth.sum() + eps)

     eyesValues.append(histBoth)
     faceValues.append(histFace)
     smileValues.append(histSmile)


neuroticismModelEyes = LinearSVR(random_state=0)
neuroticismModelFace = LinearSVR(random_state=0)
neuroticismModelSmile = LinearSVR(random_state=0)

extraversionModelEyes = LinearSVR(random_state=0)
extraversionModelFace = LinearSVR(random_state=0)
extraversionModelSmile = LinearSVR(random_state=0)

conscientiousnessModelEyes = LinearSVR(random_state=0)
conscientiousnessModelFace = LinearSVR(random_state=0)
conscientiousnessModelSmile = LinearSVR(random_state=0)

agreeablenessModelEyes = LinearSVR(random_state=0)
agreeablenessModelFace = LinearSVR(random_state=0)
agreeablenessModelSmile = LinearSVR(random_state=0)

opennessModelEyes = LinearSVR(random_state=0)
opennessModelFace = LinearSVR(random_state=0)
opennessModelSmile = LinearSVR(random_state=0)

neuroticismModelEyes.fit(eyesValues, neuroticismLabels)
neuroticismModelFace.fit(faceValues, neuroticismLabels)
neuroticismModelSmile.fit(smileValues, neuroticismLabels)

agreeablenessModelEyes.fit(eyesValues, agreeablenessLabels)
agreeablenessModelFace.fit(faceValues, agreeablenessLabels)
agreeablenessModelSmile.fit(smileValues, agreeablenessLabels)

opennessModelEyes.fit(eyesValues, opennessLabels)
opennessModelFace.fit(faceValues, opennessLabels)
opennessModelSmile.fit(smileValues, opennessLabels)

extraversionModelEyes.fit(eyesValues, extraversionLabels)
extraversionModelFace.fit(faceValues, extraversionLabels)
extraversionModelSmile.fit(smileValues, extraversionLabels)

conscientiousnessModelEyes.fit(eyesValues, conscientiousnessLabels)
conscientiousnessModelFace.fit(faceValues, conscientiousnessLabels)
conscientiousnessModelSmile.fit(smileValues, conscientiousnessLabels)


test_accuracy(opennessModelFace, rightEyeData, leftEyeData, videoLabels[0], videosData)


#
# testImage = rightEyeData[0]
#
# gray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
#
# hist1 = LBPConvertor.describe(gray)
#
# prediction = []
# prediction.append(hist1)
# print(neuroticismModel.predict(prediction))
# print(videosData['neuroticism'][videoNames[0]])
#
# cv2.destroyAllWindows()