import cv2 
import dlib 
import math
import time
from deepface import DeepFace
import matplotlib.pyplot as mp
from scipy.ndimage import uniform_filter1d
 
class Eye_Data:
    """
    class for collecting facial experessions with deep learning

    Before use, please install cv2, dlib, math and time
    """    

    #Initializing video capture, detector and predictor
    cap = cv2.VideoCapture(0)

    #Relevant variables
    list_happy = []
    list_angry = []
    list_surprise = []
    frame_data = [{}]

    #If no camera is available
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    #Do until button q is pressed in order to stop the capture
    while True:

        current_Time = time.time()

        #Capture frame-by-frame
        ret,frame=cap.read()
        if not ret:
            print("Can't receive frame")
            break
        
        #Measure facial expressions
        expression = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        for result in expression:

            #Extract happiness and anger
            emotion = result['emotion']     
            #frame_data.append({'timestamp': current_Time, 'happy': expression['happy'], 'angry': expression['angry'], 'surprise': expression['surprise']})
            list_happy.append(emotion["happy"])
            #list_angry.append(emotion["angry"])
            list_surprise.append(emotion["surprise"])

        #Show frames
        cv2.imshow("frame",frame)

        #End of capture 
        if cv2.waitKey(1) == ord("q"):
            break
    
    #Show a diagram regarding happiness
    mp.plot(list_happy, scalex = True, scaley = True)
    mp.plot(list_angry, scalex = True, scaley = True)
    mp.plot(list_surprise, scalex = True, scaley = True)

    mp.title("Measurement of Emotions")
    mp.ylabel("Percentage")
    mp.grid(True)
    mp.legend()
    mp.show()

    #End of Program
    cap.release()
    cv2.destroyAllWindows()


    
