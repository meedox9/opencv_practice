import cv2
import numpy as np

##################
# Image Practice #
##################

# img = cv2.imread('duck.png',1)
#
# img = cv2.circle(img, (255,255), 90, (0, 255, 255), -1)
# myfont = cv2.FONT_HERSHEY_COMPLEX
# img = cv2.putText(img, 'YESSIR', (0,255), myfont, 3, (255,123,123), 10, cv2.LINE_8 )
# cv2.imshow('yessir', img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##################
# Video Practice #
##################

# # 0 to enable live video camera, and path of the video for a file
# cap = cv2.VideoCapture(0)
#
# #To write video to a file in a specified codec, see: fourcc.org
# fourcc = cv2.VideoWriter_fourcc(*'MPEG') #In this example i used MPEG video codec
#
# #output options
# output = cv2.VideoWriter('gray.avi', fourcc, 20.0, (640,480)) # 20 is number of frames per seconds, i think ...
#
# #loop to capture all frames
# while(True):
#     ret, frame = cap.read()
#
#     if ret == True:
#
#         #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB) #to change video color
#         #print (cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #to get frame width, pretty usless
#
#         output.write(frame)
#         cv2.imshow('lols', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     else:
#         break
#
# cap.release()
# output.release()
# cv2.destroyAllWindows()

######################
# cascade classifier #
######################

# cap = cv2.VideoCapture(0)
# smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
#
# while(cap.isOpened()):
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2gray)
#     smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
#     for (x, y, w, h) in smiles:
#
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = frame[y:y + h, x:x + w]
#         smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
#         for (sx, sy, sw, sh) in smiles:
#             cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
#
#
#     cv2.imshow('Mbattan squad', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
face_cascade = cv2.CascadeClassifier('face.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


def detect(gray, frame):
    f_count = 0
    s_count = 0
    myfont = cv2.FONT_HERSHEY_COMPLEX
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        f_count=f_count+1
        s_count=s_count+1
        f_count_str = str(f_count)
        s_count_str = str(s_count)
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        #cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        cv2.putText(frame, 'FACE'+f_count_str, (x, y), myfont, 1, (0,0,255), 2, cv2.LINE_8 )
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 255, 0), 2)
            cv2.putText(frame, 'SMILE'+s_count_str, ((sx + sw), (sy + sh)), myfont, 1, (0,0,255), 2, cv2.LINE_8 )
    return frame

video_capture = cv2.VideoCapture(0)
while True:
   # Captures video_capture frame by frame
    _, frame = video_capture.read()

    # To capture image in monochrome
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calls the detect() function
    canvas = detect(gray, frame)

    # Displays the result on camera feed
    cv2.imshow('bruh', canvas)

    # The control breaks once q key is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the capture once all the processing is done.
video_capture.release()
cv2.destroyAllWindows()
