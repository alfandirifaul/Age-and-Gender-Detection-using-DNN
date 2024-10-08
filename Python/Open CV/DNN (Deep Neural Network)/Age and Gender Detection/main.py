# Import library and packages
import cv2 as cv
import argparse

# Function to highlight the face
def hightlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidences = detections[0, 0, i, 2]
        if confidences > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 5] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frameOpencvDnn, faceBoxes



######################## S: Global Variable
# Make object process argument from command line
parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

# Load the face model and prototxt file
faceProto = "./src/opencv_face_detector.pbtxt"
faceModel = "./src/opencv_face_detector_uint8.pb"

# Load the age model and prototxt file
ageProto = "./src/age_deploy.prototxt"
ageModel = "./src/age_net.caffemodel"

# Load the gender model and prototxt file
genderProto = "./src/gender_deploy.prototxt"
genderModel = "./src/gender_net.caffemodel"

# Define the mean values for the model
MODEL_MEAN_VALUE = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(5)', '(10)', '(20)', '(30)', '(40)', '(50)', '(60)', '(100)']
genderList = ['Male', 'Female']

# Load the model DNN
faceNet = cv.dnn.readNet(faceModel, faceProto)
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
####################### E: Global Variable

video = cv.VideoCapture(args.image if args.image else 0)
padding = 20
while cv.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv.waitKey()
        break

    resultImg, faceBoxes = hightlightFace(faceNet, frame)
    resultImg = cv.flip(resultImg, 1)
    if not faceBoxes:
        print('No Face Detected')

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
               min(faceBox[3] + padding, frame.shape[0] - 1),
               max(0, faceBox[0] - padding):
               min(faceBox[2] + padding, frame.shape[1] -1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUE,
                                    swapRB=True, crop=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print("no face detected")

        for facebox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding, frame.shape[0] - 1),
                         max(0, faceBox[0] - padding):
                         min(faceBox[2] + padding, frame.shape[1] -1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), swapRB=True, crop=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f"Gender: {gender}")

            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]
            print(f"Age: {age}")

            cv.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 255), 2, cv.LINE_AA)
            cv.imshow('Detection Result', resultImg)

cv.destroyAllWindows()
video.release()








