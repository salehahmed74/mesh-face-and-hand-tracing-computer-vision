import  cv2
import mediapipe as mp
#import numpy as np


face_Detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_Detect = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_Detect = cv2.CascadeClassifier("haarcascade_smile.xml")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
hand_draw = mp.solutions.drawing_utils


mp_faces = mp.solutions.face_mesh
faces = mp_faces.FaceMesh()
face_draw = mp.solutions.drawing_utils
drawSpecefiec = face_draw.DrawingSpec(thickness = 1, circle_radius = 1)

#vid = cv2.VideoCapture("VID saleh.mp4") #VideoCapture "function" get the path of a video with it's extension which programe can find it
vid = cv2.VideoCapture(0) # "0" means open live video from your capera

while True : # as video is consists of many images so we make a loop of images to running the video

    st , video = vid.read() #read "function" to allow programe to read video


    
    rgb_video = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)

    
    ####### Hand drawing
    hand_result = hands.process(rgb_video)
    if hand_result.multi_hand_landmarks is not None:
        for hand in hand_result.multi_hand_landmarks:
            hand_draw.draw_landmarks(video, hand, mp_hands.HAND_CONNECTIONS)



    ####### Face drawing
    face_result = faces.process(rgb_video)
    if face_result.multi_face_landmarks is not None:
        for face in face_result.multi_face_landmarks:
            face_draw.draw_landmarks(video, face, mp_faces.FACEMESH_CONTOURS,drawSpecefiec,drawSpecefiec)

        
    
##    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY) # cvtColor "function" means "convert color" it get the frame including video and call COLOR_BGR2GRAY from "cv2" to convert video into black and white


    ######### face detection
    
##    faces = face_Detect.detectMultiScale(video,1.3,5)
##
##    for (x,y,w,h) in faces:
##        cv2.rectangle(video, (x,y), (x+w,y+h),(0,250,0),2)
##        face_only = video[y:y+h,x:x+w]
##        
##    ######### eye detection        
##        eyes = eye_Detect.detectMultiScale(face_only,1.2,2)
##        for (ex,ey,ew,eh) in eyes:
##            eye_x = int ((ex+(ew/2)))-10
##            eye_y = int ((ey+(eh/2)))+10
##            cv2.putText(face_only, "X", (eye_x,eye_y), cv2.FONT_HERSHEY_COMPLEX ,1.5,(0,0,250),2)
##            
##
##    ##### smile detection        
##        smiles = smile_Detect.detectMultiScale(face_only,1.2,250)
##        for (sx,sy,sw,sh) in smiles:    
##             cv2.rectangle(face_only, (sx,sy), (sx+sw,sy+sh),(250,0,0),2)

            
    
    #print (faces)

    #size_video = cv2.resize(video,[400,700]) # resize "function" get the video, the length and the width you want to resize it to

    ######### show
    cv2.imshow("Already DEAD",video)
##    cv2.imshow("face only",face_only)

    if cv2.waitKey(28) & 0xff == ord("x"): # waitKey "function" get number of frames in each millisecond to run the video with "to caculate number of frames in each second = 1000\num_of_frames"
        break # on pressing on "x" video stop running



##image = np.zeros([400,550,190])
##cv2.line(image,[50,150],[400,100],[255,0,0],[2])
##cv2.rectangle(image,[200,150],[400,100],[255,230,0],[2])
##cv2.circle(image,[50,150],[50],[255,0,0],cv2.FILL_THICK)
##
##cv2.imshow("image",image)


##img = cv2.imread("saleh1.jpg") # imshow "function" get the path of the image with it's extension to allow program to find and read the image
##cv2.imshow("saleh",img )  # imshow "function" get title of the image where program can show it
##
##size_img = cv2.resize(img,[400,700]) # resize "function" get the image, the length and the width you want to resize it to
##crop_img = img[50:30 , 20:40]  # cropping the image
##
##cv2.imshow("saleh",size_img)



