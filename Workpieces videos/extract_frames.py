from pathlib import Path
import cv2
import numpy as np
import math
import os
import time

for tipo_workpiece in range(3):
  if tipo_workpiece == 0:
    tipo = "metal/"
  if tipo_workpiece == 1: 
    tipo = "black/"
  if tipo_workpiece == 2:
    tipo = "red/"
  folder="Workpieces videos/"
  folder = folder + tipo
  included_extensions = ['mp4','MP4']
  file_names = [fn for fn in os.listdir(folder)
                if any(fn.endswith(ext) for ext in included_extensions)]
  print(file_names)              
  # Show all images in folder
  video=0              
  for file_name in file_names:
      start = time.time()
      #print(file_name[:-4])
      video_name=folder+file_name
      cap = cv2.VideoCapture(video_name)
      images_folder='Workpieces/'+tipo
      print(video_name)
      Path(images_folder).mkdir(parents=True, exist_ok=True)

      # Check if camera opened successfully
      if (cap.isOpened()== False): 
        print("Error opening video stream or file")

      frame_num=0
      # Read until video is completed
      while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
          if frame_num % 10 == 0:   
            cv2.imwrite(images_folder+file_name+'_'+str(video)+'_'+str(frame_num)+'.jpg', frame)
          frame_num+=1 
        # Break the loop
        else: 
          break    
      # When everything done, release the video capture object
      cap.release()
      video+=1
      print("it took", time.time() - start, "seconds.")






