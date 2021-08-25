from pathlib import Path
import cv2
import numpy as np
import math
import os
import time

#folder="Workpieces videos/black/"
#folder="Workpieces videos/metal/"
folder="Workpieces videos/red/"

included_extensions = ['mp4','MP4']
file_names = [fn for fn in os.listdir(folder)
              if any(fn.endswith(ext) for ext in included_extensions)]
print(file_names)              
# Show all images in folder              
for file_name in file_names:
    start = time.time()
    #print(file_name[:-4])
    video_name=folder+file_name
    cap = cv2.VideoCapture(video_name)
    images_folder=folder+file_name[:-4]+"_frames/"
    print(images_folder)
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")

    i=0
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:   
        cv2.imwrite(images_folder+str(i)+'.jpg', frame)
        i+=1
      # Break the loop
      else: 
        break    
    # When everything done, release the video capture object
    cap.release()

    print("it took", time.time() - start, "seconds.")


