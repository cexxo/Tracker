import threading
import cv2
import featureExtractor as fe

cap=cv2.VideoCapture(0, cv2.CAP_MSMF)
global reference_img
global counter
face_match = False
global des2
global key2

def initialize(image):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

reference_img = cv2.imread("opened.jpeg")
closed = cv2.imread("closed.jpeg")
key1,des1 = fe.extract(reference_img,False)
key3,des3 = fe.extract(closed,False)

def check_face(frame):
    global face_match
    global points
    try:
        key2,des2 = fe.extract(frame,False)
        numMatches,points = fe.match(frame,reference_img,des2,des1,key2,key1,False)
        if numMatches >= 90 and not face_match:
            face_match=True
            #Gonna draw the rectangle here
        elif numMatches >= 90 and face_match:
            face_match = False
            #Gonna draw a rectangle here as well
    except ValueError:
        numMatches,points = fe.match(frame,reference_img,des2,des1,key2,key1,False)
        #pass

def recognize():
    while True:
        ret, frame = cap.read()
        if ret:
            try:
                threading.Thread(target=check_face,args=(frame.copy(),)).start()
            except ValueError:
                pass
            if face_match:
                cv2.putText(frame,"Hand open",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                #I need to understand how to calculate the points of the matches
                cv2.rectangle(frame,(int(points[-1][0]),int(points[-1][1])),(int(points[0][0]),int(points[0][1])),(0,255,0),5)
            if not face_match:
                cv2.putText(frame,"hand closed",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
                #cv2.rectangle(frame,(int(points[-1][0]),int(points[-1][1])),(int(points[0][0]),int(points[0][1])),(0,255,0),5)
            cv2.imshow("video",frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()