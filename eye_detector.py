import cv2 
import dlib 
import math
import time
 
class Eye_Data:
    """
    class for collecting data about eye movements

    Before use, please install cv2, dlib, math and time
    """

    #Useful methods
    @staticmethod
    def midpoint (p1,p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
    
    def line_length(p1,p2):
        return math.sqrt(int((p1[0] - p2[0])**2)+int((p1[1] - p2[1])**2))
    

    #Initializing video capture, detector and predictor
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #Eye parameters
    blink_count = 0
    height = 0
    start_Time = time.time()
    last_Blink_Time = start_Time
    list_Calibrate = []

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

        #Convert colored picture into gray picture
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Detect the face
        faces = detector(gray)

        #Do this for all of the detected faces
        for face in faces:                
            
            #Create the relevant landmark points
            landmarks = predictor(gray,face)
            points =[]
            for i in range(42, 48):
                points.append((landmarks.part(i).x, landmarks.part(i).y))
                cv2.circle (frame, points[i-42], 2, (0, 255, 0))

            #Create a horizontal and a vertical line for the left eye
            mid_top = midpoint(landmarks.part(44),landmarks.part(43))
            mid_bottom = midpoint(landmarks.part(47), landmarks.part(46))
            hor_line = cv2.line(frame, points[0], points[3], (0, 255, 0), 1)
            ver_line = cv2.line(frame, mid_top, mid_bottom, (0, 255, 0), 1)

            #Calibrates proximity to the screen regarding the last 3 seconds
            trigger = line_length(mid_bottom, mid_top)
            if current_Time-start_Time < 3:
                list_Calibrate.append(trigger)
            else:
                list_Calibrate.pop(0)
                list_Calibrate.append(trigger)
                height = sum(list_Calibrate)/len(list_Calibrate)

            #Blinking trigger 
            if trigger < height*0.8 and (time.time()-last_Blink_Time)>0.5 and time.time()-start_Time>3: 
                blink_count = blink_count + 1
                last_Blink_Time = time.time()
            
            #Show blinking counter
            cv2.putText(frame, "blink_count:"+str(blink_count), (100, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

        #Show frames
        cv2.imshow("frame",frame)

        #End of capture 
        if cv2.waitKey(1) == ord("q"):
            break

    #End of Program
    cap.release()
    cv2.destroyAllWindows()


    
