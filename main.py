import newRecognition as nr
import Recognition as r
import featureExtractor as fe
import cv2

def main():
    open = cv2.imread("open.jpeg")
    closed = cv2.imread("closed.jpeg")
    #key1,des1=fe.extract(image,False)
    #key2,des2=fe.extract(image,False)
    #fe.match(image,image,des1,des2,key1,key2,False)
    #r.initialize(open)
    #r.recognize()
    nr.recognize(open,closed)


if __name__ == "__main__":
    main()