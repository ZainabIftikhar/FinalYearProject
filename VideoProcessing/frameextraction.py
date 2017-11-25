import cv2


class FrameExtraction:
    def __init__(self, videoCount, interval, videoNames, folderPath):
        self.videoCount = videoCount
        self.interval = interval
        self.videoNames = videoNames
        self.folderPath = folderPath

    def extract(self):
        # loading the haarcascade XML files to detect facial features
        face_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_frontalface_default.xml')
        left_eye_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_lefteye_2splits.xml')
        right_eye_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_righteye_2splits.xml')
        smile_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_smile.xml')

        facesData = []
        leftEyeData = []
        smileData = []
        rightEyeData = []
        faceLabels = []
        smileVideoLabels = []
        rightEyeLabels = []
        leftEyeLabels = []
        loopbreak = False
        videosLoaded = 0
        print(len(self.videoNames))
        for videoName in self.videoNames:
            if loopbreak:
                print("Loop broken")
                break

            # calculating the path of the file to load next
            # videoName = list(labelDictionary.keys())[i]

            # Loading the video
            capVideo = cv2.VideoCapture(self.folderPath + videoName)

            framePosition = 0

            if capVideo.isOpened():
                videosLoaded += 1
                print('Processing video no: {} with filename: {}'.format(videosLoaded, videoName))
                if videosLoaded == self.videoCount:
                    loopbreak = True
            else:
                print('Video with filename: {} not found.'.format(videoName))
            while capVideo.isOpened():

                # setting the time of the video to next frame
                if capVideo.set(cv2.CAP_PROP_POS_MSEC, framePosition):
                    # print("condition met")

                    # adding 100ms to next frame position
                    framePosition += self.interval

                    # getting frames from the video
                    ret, frame = capVideo.read()

                    if frame is None:
                        break
                    else:
                        height, width, layers = frame.shape
                        frame = cv2.resize(frame, (int(width / 4), int(height / 4)))
                        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # Detecting the face
                        faces = face_cascade.detectMultiScale(grayFrame, 1.3, 5)

                        for (x, y, w, h) in faces:

                            # cropping both original frame and the gray frame
                            croppedFace = frame[y:y + h, x:x + h]
                            croppedFaceGray = grayFrame[y:y + h, x:x + w]

                            # detecting both left and right
                            leftEye = left_eye_cascade.detectMultiScale(croppedFaceGray)  # Detecting left eye
                            rightEye = right_eye_cascade.detectMultiScale(croppedFaceGray)  # Detecting right eye
                            smile = smile_cascade.detectMultiScale(croppedFaceGray)

                            for (ex, ey, ew, eh) in leftEye:
                                # cv2.rectangle(croppedFace, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
                                leftEyeFrame = croppedFace[ey:ey + eh, ex:ex + ew]
                                leftEyeData.append(leftEyeFrame)
                                leftEyeLabels.append(videoName)
                                break
                            for (ex, ey, ew, eh) in rightEye:
                                # cv2.rectangle(croppedFace, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                                rightEyeFrame = croppedFace[ey:ey + eh, ex:ex + ew]
                                rightEyeData.append(rightEyeFrame)
                                rightEyeLabels.append(videoName)
                                break
                            for (sx, sy, sw, sh) in smile:
                                # cv2.rectangle(croppedFace, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
                                smileFrame = croppedFace[sy:sy + sh, sx:sx + sw]
                                smileData.append(smileFrame)
                                smileVideoLabels.append(videoName)
                                break

                            facesData.append(croppedFace)
                            faceLabels.append(videoName)
                            break
                else:
                    # video reaches its end
                    capVideo.release()
                    break

        return facesData, leftEyeData, rightEyeData, smileData, faceLabels, leftEyeLabels, rightEyeLabels, smileVideoLabels

    def extract_single(self, videoName):
        face_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_frontalface_default.xml')
        left_eye_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_lefteye_2splits.xml')
        right_eye_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_righteye_2splits.xml')
        smile_cascade = cv2.CascadeClassifier('HaarCascadeFiles/haarcascade_smile.xml')

        facesData = []
        leftEyeData = []
        smileData = []
        rightEyeData = []

        capVideo = cv2.VideoCapture(self.folderPath + videoName)

        framePosition = 0

        if capVideo.isOpened():
            print('Processing video with filename: {}'.format(videoName))
        else:
            print('Video with filename: {} not found.'.format(videoName))
        while capVideo.isOpened():
            # setting the time of the video to next frame
            if capVideo.set(cv2.CAP_PROP_POS_MSEC, framePosition):
                # print("condition met")

                # adding 100ms to next frame position
                framePosition += self.interval

                # getting frames from the video
                ret, frame = capVideo.read()

                if frame is None:
                    break
                else:
                    height, width, layers = frame.shape
                    frame = cv2.resize(frame, (int(width / 4), int(height / 4)))
                    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detecting the face
                    faces = face_cascade.detectMultiScale(grayFrame, 1.3, 5)

                    for (x, y, w, h) in faces:

                        # cropping both original frame and the gray frame
                        croppedFace = frame[y:y + h, x:x + h]
                        croppedFaceGray = grayFrame[y:y + h, x:x + w]

                        # detecting both left and right
                        leftEye = left_eye_cascade.detectMultiScale(croppedFaceGray)  # Detecting left eye
                        rightEye = right_eye_cascade.detectMultiScale(croppedFaceGray)  # Detecting right eye
                        smile = smile_cascade.detectMultiScale(croppedFaceGray)

                        for (ex, ey, ew, eh) in leftEye:
                            # cv2.rectangle(croppedFace, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
                            leftEyeFrame = croppedFace[ey:ey + eh, ex:ex + ew]
                            leftEyeData.append(leftEyeFrame)
                            break
                        for (ex, ey, ew, eh) in rightEye:
                            # cv2.rectangle(croppedFace, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                            rightEyeFrame = croppedFace[ey:ey + eh, ex:ex + ew]
                            rightEyeData.append(rightEyeFrame)
                            break
                        for (sx, sy, sw, sh) in smile:
                            # cv2.rectangle(croppedFace, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
                            smileFrame = croppedFace[sy:sy + sh, sx:sx + sw]
                            smileData.append(smileFrame)
                            break
                        facesData.append(croppedFace)
                        break
            else:
                # video reaches its end
                capVideo.release()
                break

        return facesData, smileData, leftEyeData, rightEyeData

