import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import filedialog as fd


def opticalflow(frames):
    #define variables and arrays
    width,height,c = frames[0].shape#define widh and height for crop and resize
    shiftings_x = []
    shiftings_y = []
    shiftings = []

    #paramater of flann match
    FLAN_INDEX_KDTREE = 0
    index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params) #define flann matcher
    sift = cv2.xfeatures2d.SIFT_create()#define sift for detecting features
    output = [] #output array 
    border = 80 #border valur for complete the missing areas
    orj_frames = frames #temp array for reaching unchanged frames

    #loop for applying all frames 
    for i in range(len(frames)-1):
        oldframe = frames[i]
        newframe = frames[i+1]

        #detect keypoints
        keypoints1, descriptors1 = sift.detectAndCompute(oldframe, None)
        keypoints2, descriptors2 = sift.detectAndCompute(newframe, None)
        matches = flann.knnMatch (descriptors1, descriptors2,k=2)
        
        good_matches = [] #array for choose good matches
        matched_keypoint_locations = [] #array for hold features locations
        #get the best matches
        for m1, m2 in matches:
            if m1.distance < 0.5 * m2.distance:
                good_matches.append([m1])
                x1, y1 = keypoints1[m1.queryIdx].pt
                x2, y2 = keypoints2[m1.trainIdx].pt
                #points locations array
                matched_keypoint_locations.append(((int(x1), int(y1)), (int(x2), int(y2))))


        """#show the matched features
        flann_matches =cv2.drawMatchesKnn(oldframe, keypoints1, newframe, keypoints2, good_matches, None, flags=2)
        cv2.imshow("flann",flann_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        #array to hold values of vectors
        keypoint_x_mag = []
        keypoint_y_mag = []
        keypoint_x_sum = 0
        keypoint_y_sum = 0

        #get magitudes and sum the distance between matched keypoints
        for point in matched_keypoint_locations:
            point1x = point[0][0]
            point2x = point[1][0]
            keypoint_x_mag.append(point2x-point1x)
            keypoint_x_sum += (point2x-point1x)

            point1y = point[0][1]
            point2y = point[1][1]
            keypoint_y_mag.append(point2y-point1y)
            keypoint_y_sum += (point2y-point1y)
        
        #calculate avarage of distance for x and y axis
        keypoint_x_avg = keypoint_x_sum / len(matched_keypoint_locations)
        keypoint_y_avg = (keypoint_y_sum / len(matched_keypoint_locations))

        #elimination of off-mean values
        thresholdx = keypoint_x_avg*0.8
        thresholdy = keypoint_y_avg*0.8
        print("avarage x movement:",round(keypoint_x_avg),"    avarage y movement:",round(keypoint_y_avg))
        print(len(matched_keypoint_locations),"elements before elimination ")
        print("thresholds x,y: ",int(keypoint_x_avg+thresholdx),"-",int(keypoint_x_avg-thresholdx),",",int(keypoint_y_avg+thresholdy),"-",int(keypoint_y_avg-thresholdy))
        for point in matched_keypoint_locations:
            point1x = point[0][0]
            point2x = point[1][0]
            point1y = point[0][1]
            point2y = point[1][1]
            diffx = point[1][0] - point[0][0]
            diffy = point[1][1] - point[0][1]
            if keypoint_x_avg < 0:
                if keypoint_y_avg <0:
                    if diffx < keypoint_x_avg + thresholdx or diffx >= keypoint_x_avg - thresholdx or diffy < keypoint_y_avg + thresholdy or diffy >= keypoint_y_avg - thresholdy:
                        matched_keypoint_locations.remove(point)
                        keypoint_x_mag.remove(diffx)
                        keypoint_y_mag.remove(diffy)
                        keypoint_x_sum  = keypoint_x_sum - diffx
                        keypoint_y_sum  = keypoint_y_sum - diffy
                else:
                    if diffx < keypoint_x_avg + thresholdx or diffx >= keypoint_x_avg - thresholdx or diffy > keypoint_y_avg + thresholdy or diffy <= keypoint_y_avg - thresholdy:
                        matched_keypoint_locations.remove(point)
                        keypoint_x_mag.remove(diffx)
                        keypoint_y_mag.remove(diffy)
                        keypoint_x_sum  = keypoint_x_sum - diffx
                        keypoint_y_sum  = keypoint_y_sum - diffy
            elif keypoint_x_avg >=0:
                if keypoint_y_avg <0:
                    if diffx > keypoint_x_avg + thresholdx or diffx <= keypoint_x_avg - thresholdx or diffy < keypoint_y_avg + thresholdy or diffy >= keypoint_y_avg - thresholdy:
                        matched_keypoint_locations.remove(point)
                        keypoint_x_mag.remove(diffx)
                        keypoint_y_mag.remove(diffy)
                        keypoint_x_sum  = keypoint_x_sum - diffx
                        keypoint_y_sum  = keypoint_y_sum - diffy
                else:
                    if diffx > keypoint_x_avg + thresholdx or diffx <= keypoint_x_avg - thresholdx or diffy > keypoint_y_avg + thresholdy or diffy <= keypoint_y_avg - thresholdy:
                        matched_keypoint_locations.remove(point)
                        keypoint_x_mag.remove(diffx)
                        keypoint_y_mag.remove(diffy)
                        keypoint_x_sum  = keypoint_x_sum - diffx
                        keypoint_y_sum  = keypoint_y_sum - diffy   

        print(len(matched_keypoint_locations),"elements after elimination ")

        """#plot the magnitude of distance between features
        plt.subplot(211)
        plt.plot(keypoint_x_mag, marker="o",label="X değerleri")
        
        plt.subplot(212)
        plt.plot(keypoint_y_mag,marker="o",label = "y değerleri")
        plt.show()"""
        #re-calculate avarage
        keypoint_x_avg = keypoint_x_sum / len(matched_keypoint_locations)
        keypoint_y_avg = (keypoint_y_sum / len(matched_keypoint_locations))
        
        #assing shifting value
        shiftx = -int(round(keypoint_x_avg))
        shifty = -int(round(keypoint_y_avg)) 
        shiftings_x.append(abs(shiftx))
        shiftings_y.append(abs(shifty))
        print("shiftx,y",shiftx,shifty)
        
        """
        if shiftx > border :
            shiftx = border
        elif shiftx < -border:
            shiftx = -border
        elif shifty > border:
            shifty = border
        elif shifty < -border:
            shifty = -border"""

        height, width = newframe.shape[:2]

        T = np.float32([[1, 0, shiftx], [0, 1, shifty]])

        #assigning the values of the fields to be filled and crop 
        if shiftx >= 0 :
            left = shiftx
            right = width
            fill_right = 0
            fill_left = left
            crop_left = 0
            crop_right = shiftx
            
        
        elif shiftx < 0:
            left = 0
            right = width + shiftx
            fill_right = -shiftx
            fill_left = left
            crop_left = width + shiftx
            crop_right = width
    
        if shifty >= 0:
            top = shifty
            bot = height
            fill_top = top
            fill_bot = 0
            crop_top = 0
            crop_bot = shifty
            crop_top_x = shifty
            crop_bot_x = height

        elif shifty < 0:
            bot = height + shifty
            top = 0
            fill_top = 0
            fill_bot = -shifty
            crop_top = height + shifty
            crop_bot = height
            crop_top_x = 0
            crop_bot_x = height+shifty
            
        
        

        #hconcat for x axis , vconcat with y axis
        #shift the image
        img_translation = cv2.warpAffine(newframe, T, (width , height ))
        #cropping shifted regions
        cropped_img = img_translation[top:bot,left:right]


        #filling the shifted blank region with prev frame's 
        if shiftx != 0 and shifty == 0:
            prevx = orj_frames[i][0:height,crop_left:crop_right]
            if shiftx > 0:
                filled_img = cv2.hconcat((prevx,cropped_img))
            else:
                filled_img = cv2.hconcat((cropped_img,prevx))
            
        elif shiftx == 0 and shifty != 0:
            prevy = orj_frames[i][crop_top:crop_bot,0:width]
            if shifty > 0:
                filled_img = cv2.vconcat((prevy,cropped_img))
            else:
                filled_img = cv2.vconcat((cropped_img,prevy))
            
        elif shiftx != 0 and shifty != 0:
            prevx = orj_frames[i][crop_top_x:crop_bot_x,crop_left:crop_right]
            prevy = orj_frames[i][crop_top:crop_bot,0:width]
            if shiftx > 0 and shifty >0:
                filled_img = cv2.hconcat((prevx,cropped_img))
                filled_img = cv2.vconcat((prevy,filled_img))
            elif shiftx > 0 and shifty < 0:
                filled_img = cv2.hconcat((prevx,cropped_img))
                filled_img = cv2.vconcat((filled_img,prevy))
            elif shiftx < 0 and shifty >0:
                filled_img = cv2.hconcat((cropped_img,prevx))
                filled_img = cv2.vconcat((prevy,filled_img))
            elif shiftx < 0 and shifty < 0:
                filled_img = cv2.hconcat((cropped_img,prevx))
                filled_img = cv2.vconcat((filled_img,prevy))
                
        else:
            #filling shifted region with border replicate if there is an problem
            filled_img = cv2.copyMakeBorder(cropped_img, fill_top, fill_bot, fill_left, fill_right, cv2.BORDER_REPLICATE)

        
        #resized_image = cv2.resize(cropped_img,(width,height))
        frames[i+1] = filled_img
        shiftings.append((shiftx,shifty))
        output.append(filled_img)
    return output

#function for reading frame of a video
def framesFromVid(path):
    readed_frames = []
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    while success:
        readed_frames.append(image) 
        success,image = vidcap.read()
        count += 1
    print("readed frames :",count)
    return readed_frames

#function for cropping images 
def cropImages(frames,shiftings):
    count = 0 
    outputframes = []
    width,height,c = frames[0].shape
    shiftx = max(shiftings[0])
    shiftx_sum = sum(shiftings[0])
    shiftx_avg = int(shiftx_sum / len(shiftings[0]))
    shifty = max(shiftings[1])
    shifty_sum = sum(shiftings[1])
    shifty_avg = int(shifty_sum / len(shiftings[1]))
    
    if shiftx_avg >=0:
        left = int(shiftx_avg/2)
        right = width-int(shiftx_avg/2)
    elif shiftx_avg < 0:
        left = abs(int(shiftx_avg/2))
        right = width - abs(int(shiftx_avg/2))

    if shifty_avg >= 0:
        bot = int(shifty_avg/2)
        top = height - int(shifty_avg/2)
    elif shifty_avg < 0:
        bot = abs(int(shifty_avg/2))
        top = height - abs(int(shifty_avg/2))

    for frame in frames:
          
        cropped_img = frame[bot:top,left:right]

        outputframes.append(cropped_img)

    return outputframes

video_path = fd.askopenfilename()
video_name = video_path.split("/")[-1].split(".")[0]
print(video_name)

#read frames
frames = framesFromVid(video_path)

#send frames to optical flow function for detect features
#match features , shift frames opposite of movement and fill the blank areas
shiftedframes = opticalflow(frames)
h,w,c = shiftedframes[0].shape
size = w,h #get dimensions for video same size as frames
#define videowriter
video = cv2.VideoWriter('test'+video_name+'.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30.0, size)
#write the shifted frames to a video
for i in range(len(shiftedframes)):
    video.write(shiftedframes[i])
video.release()
cv2.destroyAllWindows()
for img in shiftedframes:
    cv2.imshow('frame',img )
    cv2.waitKey(0)


cv2.destroyAllWindows()

