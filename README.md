# Video stabilization with optical flow algorithm
Video stabilization with optical flow algorithm is an image processing technique that aims to create a stable video by tracking the movement of a video frame. 
The Video Stabilization with Optical Flow algorithm developed during my internship is explained step by step.
The input video is selected and the frames in the video are taken. Then, feature detection is performed on consecutive frames with the feature detection algorithm (sift feature detection) and a series of features are obtained. The features between the two frames obtained are matched with each other (with flat face match). A more stable matching result is obtained by eliminating the matches that are far from a certain threshold value from the results obtained as a result of matching. The pixel coordinates of the matched features are taken and the distances to each other are obtained by subtracting the x and y coordinates of both pixels. All the distance values ​​obtained are added to obtain an average motion value. This motion value is the value that will be used for bringing the two frames closer to each other and fixing them. The second frame that comes after is shifted in a way opposite to this value and the two moving frames are made more stable.

I added a file selection screen that opens to receive the input video and by sending the selected file to the separation function that I wrote to the frames, a series containing the frames is obtained as feedback. Afterwards, the frames are sent in series to the “opticalflow” function, which is the main function where all operations will be performed.

'''python
video_path = fd.askopenfilename() 
video_name = video_path.split("/")[-1].split(".")[0] 
print(video_name) 
#read frames 
frames = framesFromVid(video_path) 
#send frames to optical flow function for detect features 
#match features , shift frames opposite of movement and fill the blank areas
'''
