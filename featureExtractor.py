import cv2

def extract(image,flag):
    copy=image.copy()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image,None)
    siftImage = cv2.drawKeypoints(image,keypoints,copy)
    if flag:
        cv2.namedWindow("features",cv2.WINDOW_NORMAL)
        cv2.imshow("features",siftImage)
        cv2.waitKey(0)
    return keypoints,descriptors

def getExtremes(image,target,desImage,desTarget,keyImage,keyTarget):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches=bf.match(desImage,desTarget)
    matches = sorted(matches, key=lambda x:x.distance)
    matchImg = cv2.drawMatches(image,keyImage,target,keyTarget,matches,target,flags=2)
    return matches

def match(image,target,desImage,desTarget,keyImage,keyTarget,flag):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches=bf.match(desImage,desTarget)
    matches = sorted(matches, key=lambda x:x.distance)
    matchImg = cv2.drawMatches(image,keyImage,target,keyTarget,matches,target,flags=2)
    #print(matches[0].trainIdx)
    if flag:
        cv2.namedWindow("matcher",cv2.WINDOW_NORMAL)
        cv2.imshow("matcher",matchImg)
        cv2.waitKey(0)
        #####
    list_kp1 = []
    list_kp2 = []

    # For each match...
    for mat in matches[:5]:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = keyImage[img1_idx].pt
        (x2, y2) = keyTarget[img2_idx].pt
        

        # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))



    list_kp1 = [keyImage[mat.queryIdx].pt for mat in matches] 
    list_kp2 = [keyTarget[mat.trainIdx].pt for mat in matches]
        #####
    return len(matches),list_kp1