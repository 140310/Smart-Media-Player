# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import time 
from imutils.video import VideoStream
from imutils.video import FPS
# construct the argument parser and parse the arguments
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
COUNTER = 0


data_path="./Video/Rayan.mp4"
#video = Video(data_path)
#video_capture = cv2.VideoCapture(0) # reading frames
#hoststr = "./Video/Moiz.mp4"
cap = VideoStream (src = 0).start()
time.sleep (2.0)
fps = FPS().start()



#success,image = video_capture.read()
count = 0
#success = True
totalTime = time.time()
while True:
#generator = np.arange(video.start, video.end, video.step*4)
#for t in generator:
#    start=time.time()
#    image = video._get_frame(t)
    
    image = cap.read()

    if image is None:
 #       print ("Time required to process a fullVideo " ,time.time()-totalTime)
        break

    # load the input image, resize it, and convert it to grayscale
    if  count%4==0:
        image = imutils.resize(image, width=200)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        #start1=time.time()
        #print gray.shape
        rects = detector(gray, 1)
        #print ("Time required to detectface" ,time.time()-start1)
        clone = image.copy()
        # loop over the face detections
        found=False
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    print count,"Paused Due to eye closed"
            else:
                COUNTER = 0
                
            # loop over the face parts individually

            i,j=36,48
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            
            found=True
            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                # extract the ROI of the face region as a separate image
        if not found:
            print count," Paused due to not watching "
        cv2.imshow("Image", clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    count+=1
#    print ("Time required to process a frame" ,time.time()-start)
#print ("Time required to process a fullVideo " ,time.time()-totalTime)
