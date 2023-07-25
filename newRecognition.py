import threading
import cv2
import featureExtractor as fe

WINDOW_WIDTH = 480
WINDOW_HEIGHT= 320

cap=cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

def checkHandOpen(frame,referenceImg,key1,des1):
    handOpen = False
    try:
        #key1,des1 = fe.extract(referenceImg,False)
        key2,des2 = fe.extract(frame,False)
        numMatches,points = fe.match(frame,referenceImg,des2,des1,key2,key1,False)
        if numMatches >= 80 and not handOpen:
            handOpen=True
    except ValueError:
        numMatches,points = fe.match(frame,referenceImg,des2,des1,key2,key1,False)
    return handOpen

def checkHandClosed(frame,referenceImg,key1,des1):
    handClosed = False
    try:
        #key1,des1 = fe.extract(referenceImg,False)
        key2,des2 = fe.extract(frame,False)
        numMatches,points = fe.match(frame,referenceImg,des2,des1,key2,key1,False)
        if numMatches >= 60 and not handClosed:
            handClosed=True
    except ValueError:
        numMatches,points = fe.match(frame,referenceImg,des2,des1,key2,key1,False)
    return handClosed

def recognize(open,closed):
    key1,des1 = fe.extract(open,False)
    key3,des3 = fe.extract(closed,False)
    coordinates = []
    while True:
        offsetX = 0 #it was 300
        offsetY = 0 #it was 75
        ret, frame = cap.read()
        key2,des2 = fe.extract(frame,False)
        key2x = sorted(key2, key=lambda x:x.pt[0])
        key2y = sorted(key2, key=lambda x:x.pt[1])
        #key2y = sorted(key2, key=lambda x:key2[x].pt[1].distance)
        if ret:
            try:
                threading.Thread(target=checkHandOpen,args=(frame.copy(),open,key1,des1,)).start()
            except ValueError:
                pass
            """try:
                threading.Thread(target=checkHandClosed,args=(frame.copy(),closed,key1,des1,)).start()
            except ValueError:
                pass"""
            if checkHandOpen(frame,open,key1,des1):
                cv2.putText(frame,"Hand open",(int(key2x[0].pt[0])+offsetX,int(key2y[-1].pt[1]-offsetY)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
                try:
                    #print("im drawing")
                    #cv2.rectangle(frame,(int(key1[0].pt[0])+600,int(key1[-1].pt[1])),(int(key1[0].pt[0])+100,int(key1[0].pt[1])-200),(0,255,0),3)
                    cv2.rectangle(frame,(int(key2x[0].pt[0])+offsetX,int(key2y[0].pt[1])+offsetY),(int(key2x[-1].pt[0])+offsetX,int(key2y[-1].pt[1])),(0,255,0),3)
                except ValueError:
                    pass
                #I need to understand how to calculate the points of the matches
                #cv2.rectangle(frame,(int(points[-1][0]),int(points[-1][1])),(int(points[0][0]),int(points[0][1])),(0,255,0),5)
            elif checkHandClosed(frame,closed,key3,des3):
                cv2.putText(frame,"hand closed",(int(key2x[0].pt[0])+offsetX,int(key2y[-1].pt[1]-offsetY)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
                cv2.rectangle(frame,(int(key2x[0].pt[0])+offsetX,int(key2y[0].pt[1])+offsetY),(int(key2x[-1].pt[0])+offsetX,int(key2y[-1].pt[1])),(0,0,255),3)
                #cv2.rectangle(frame,(int(key3[-1].pt[0]),int(key3[-1].pt[1])),(int(key3[0].pt[0]),int(key3[0].pt[1])),(0,0,255),3)
                #cv2.rectangle(frame,(int(points[-1][0]),int(points[-1][1])),(int(points[0][0]),int(points[0][1])),(0,255,0),5)
            else:
                cv2.putText(frame,"No hand detected",(20,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            cv2.imshow("video",frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()