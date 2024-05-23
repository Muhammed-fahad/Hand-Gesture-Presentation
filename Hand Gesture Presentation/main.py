import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

width, height = 640, 480
folderPath = "presentation"

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get and sort presentation images
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
pathImages = sorted([f for f in os.listdir(folderPath) if f.lower().endswith(valid_extensions)], key=len)
#print(pathImages)

# Variables
imgNumber = 0
hs, ws = int(120 * 1), int(213 * 1)
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = -1
annotationStart = False
smoothFactor = 5  # Number of frames to average for smoothing

# List to store recent index finger positions for smoothing
recentPoints = []

# Hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Define the active region for hand movements (middle half of the screen)
activeRegionXMin = width // 2
activeRegionXMax = 3 * width // 4
activeRegionYMin = height // 4
activeRegionYMax = 3 * height // 4

def interpolate(value, old_min, old_max, new_min, new_max):
    return int(np.interp(value, [old_min, old_max], [new_min, new_max]))

def smoothPoints(points, smoothFactor):
    if len(points) < smoothFactor:
        return points[-1]
    else:
        avgX = int(np.mean([p[0] for p in points[-smoothFactor:]]))
        avgY = int(np.mean([p[1] for p in points[-smoothFactor:]]))
        return (avgX, avgY)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    # Check if there are images in the folder
    if not pathImages:
        print("No images found in the folder.")
        break

    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Check if the image is loaded successfully
    if imgCurrent is None:
        print(f"Failed to load image: {pathFullImage}")
        break

    hands, img = detector.findHands(img)

    if hands and not buttonPressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand["center"]
        lmList = hand['lmList']

        # Interpolate index finger position to the full image dimensions
        xVal = interpolate(lmList[8][0], activeRegionXMin, activeRegionXMax, 0, imgCurrent.shape[1])
        yVal = interpolate(lmList[8][1], activeRegionYMin, activeRegionYMax, 0, imgCurrent.shape[0])
        indexFinger = (xVal, yVal)

        # Append the current position to recent points
        recentPoints.append(indexFinger)
        
        # Smooth the index finger position
        smoothedIndexFinger = smoothPoints(recentPoints, smoothFactor)

        if cy <= gestureThreshold:
            if fingers == [1, 0, 0, 0, 0]:
                #print("Left")
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber -= 1

            if fingers == [0, 0, 0, 0, 1]:
                #print("Right")
                if imgNumber < len(pathImages) - 1:
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    buttonPressed = True
                    imgNumber += 1

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, smoothedIndexFinger, 12, (0, 0, 255), cv2.FILLED)
        
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
                
            cv2.circle(imgCurrent, smoothedIndexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(smoothedIndexFinger)
        else:
            annotationStart = False
            
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True
                
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

    # Adding webcam in slide
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("n"):  # Go to next slide on 'n' key press
        imgNumber = (imgNumber + 1) % len(pathImages)
    elif key == ord("p"):  # Go to previous slide on 'p' key press
        imgNumber = (imgNumber - 1 + len(pathImages)) % len(pathImages)

cap.release()
cv2.destroyAllWindows()