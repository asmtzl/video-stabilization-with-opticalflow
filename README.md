# Video stabilization with optical flow algorithm
Video stabilization with optical flow algorithm is an image processing technique that aims to create a stable video by tracking the movement of a video frame. 
The Video Stabilization with Optical Flow algorithm developed during my internship is explained step by step.
The input video is selected and the frames in the video are taken. Then, feature detection is performed on consecutive frames with the feature detection algorithm (sift feature detection) and a series of features are obtained. The features between the two frames obtained are matched with each other (with flat face match). A more stable matching result is obtained by eliminating the matches that are far from a certain threshold value from the results obtained as a result of matching. The pixel coordinates of the matched features are taken and the distances to each other are obtained by subtracting the x and y coordinates of both pixels. All the distance values ​​obtained are added to obtain an average motion value. This motion value is the value that will be used for bringing the two frames closer to each other and fixing them. The second frame that comes after is shifted in a way opposite to this value and the two moving frames are made more stable.
## Input video
I added a file selection screen that opens to receive the input video and by sending the selected file to the separation function that I wrote to the frames, a series containing the frames is obtained as feedback. Afterwards, the frames are sent in series to the “opticalflow” function, which is the main function where all operations will be performed.

```python
video_path = fd.askopenfilename() 
video_name = video_path.split("/")[-1].split(".")[0] 
print(video_name) 
#read frames 
frames = framesFromVid(video_path) 
#send frames to optical flow function for detect features 
#match features , shift frames opposite of movement and fill the blank areas
shiftedframes = opticalflow(frames) 
```
## Extract frames
The OpenCV library is used to extract the frames in the video and the frames are transferred to an array with the following function.
```python
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
```

In order to calculate the value to be shifted between frames, feature detection process is required between consecutive frames, for this purpose SIFT algorithm in OpenCV library is used. The reason for using SIFT algorithm is that it is more effective than SURF algorithm. Articles analyzing and comparing two algorithms were examined and this decision was reached as a result.

FLANN algorithm was used to match detected features with each other. For more stable and efficient results in matching, unstable matches, i.e. matches with high distance values, are eliminated based on the distance values ​​of the matched features. Thus, more stable matches closer to accuracy are obtained. As can be seen in Figure 3.2.13 below, the effect of change in distance value on matches is given with an example. It can be observed that deviant matches increase visibly when the threshold is increased. It is seen that the accuracy rate of matches increases as the distance value decreases. Stable and most accurate matches were selected as a result of elimination with the most optimum threshold value by neither eliminating too many matches nor keeping too many matches.
## Shifting the frames
These matches are used to calculate which way and how much the movement between two frames has moved. The coordinate values ​​of the matching features in the new and old frames are subtracted from each other to obtain how much and which way they have moved in the x and y coordinates, and the amount of movement found for each feature is added, then an average movement value is calculated.

In order to bring the average movement value closer to accuracy, to obtain a stable value, and to eliminate moving or incorrectly matched values ​​in the frames, a selection is made again. This selection is made by eliminating values ​​that are a certain amount below and above the average value, that is, values ​​that are very far from the average. Two different threshold values ​​are calculated for selection, more than 80% or less than 20% of the average value, and the movement value of each match is checked according to these threshold values. Values ​​that are not in this range are deleted from the arrays where the matches are located, and finally the average is calculated again with stable values.

As a result of these operations, we calculate the movement values ​​between each frame. The code sections where all these steps are performed are as follows.

After calculating the motion vectors in the frames and calculating how much the frames will be shifted in which direction, the frames are shifted by calling the OpenCV ‘warpAffine’ function to perform the shifting operation. The necessary parameters must be provided for this function to perform the shifting operation. For this, the shift data is assigned to a matrix and sent to the function as a parameter.
```python
 T = np.float32([[1, 0, shiftx], [0, 1, shifty]]) 
        #shift the image 
        img_translation = cv2.warpAffine(newframe, T, (width , height )) 
```
After the scrolling process, black areas are formed on the scrolled sides of the frames. Different approaches such as filling or cutting these black areas have been used in different solutions. These areas can be cut equally from all sides and the video can be saved in a reduced size. Based on the articles I have researched and read, original solutions have been tried to be produced to solve this problem.
A more natural image was attempted by cutting the cut parts from the original unshifted frames after each shift and adding them to the black areas. For this, the shifted black areas and the shifted areas in the old frames were cut and added using the data on the top that holds how much shift was made from which regions. The ‘vconcat’ function in OpenCV was used for the addition process, as in the code above.
