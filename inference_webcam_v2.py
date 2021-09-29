from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os
import time


model_name = "MobileNetV2"
model=load_model(model_name+'/model.h5')


def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    detections=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)       
        if area>100 and area<1000:
            cv2.drawContours(imgContour,cnt,-1,(0,255,0),1)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)        
            objCorner = len(approx)
            x,y,w,h=cv2.boundingRect(approx)
            if h>=3*w or w>=3*h:    
                pass
            else:    
                cv2.rectangle(imgContour,(x,y),(x+w,y+h),(255,0,0),1)        
                detections.append([x,y,w,h])
    return detections

def detect(img,detections):    
    num=0
    detects=[]
    a=0
    for [x,y,w,h] in detections:
        if w <=25 and h <=25:            
            min_hor=int(0.99*x)
            max_hor=int(1.01*(x+w))
            min_ver=int(0.99*y)
            max_ver=int(1.01*(y+h))
            region=img[min_ver:max_ver,min_hor:max_hor]
            region=cv2.resize(region, (w*5,h*5), interpolation = cv2.INTER_AREA)
            imgGray = cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
            size_blur=3    
            circles1 =  cv2.HoughCircles(imgGray,cv2.HOUGH_GRADIENT,1,minDist=1000,param1=100,param2=20,minRadius=1,maxRadius=20)
            
            if circles1 is not None:
                circles1 = np.round(circles1[0, :]).astype("int")

                for _ in circles1:
                    img_ = image.array_to_img(region)
                    # img = region
                    img_ = img_.resize((224, 224))

                    x_=image.img_to_array(img_)
                    x_=x_/255
                    x_=np.expand_dims(x_,axis=0)

                    y_pred=model.predict(x_)

                    resultado = int(np.argmax(y_pred))

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (x,y)
                    fontScale = 1
                    lineType = 2
                    result_text = "indefinido"

                    if resultado == 1:
                        fontColor = (0,255,0)
                        result_text = "Conforme"
                    else:
                        fontColor = (0,0,255)
                        result_text = "Nao conforme"


                    cv2.putText(imgContour,result_text, 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)

                    a+=1              

def getArea(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    detections=[]
    max_area = 0
    #print(len(contours))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>max_area:
            max_area = area
            peri = cv2.arcLength(cnt,True)
            contour=cnt
    approx = cv2.approxPolyDP(contour,0.02*peri,True)        
    objCorner = len(approx)
    #get bounding box coordinates, width and height
    x,y,w,h=cv2.boundingRect(approx)
    #print(x,y,w,h)
    # cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),1)        
    detections.append([x,y,w,h])
    return max_area


modo = int(input("0 Para utilizar video\n1 Para utiliza Webcam\n"))

# A cada quantos frames a IA irá realizar a análise
fps_limit = int(input("Realizar a análise a cada quantos frames? [1 = analise em todo frame] "))
# fps_limit = 1

# Setup com video
if modo == 0:
    folder="./"
    included_extensions = ['mp4','MP4','MOV']
    file_names = [fn for fn in os.listdir(folder)
                if any(fn.endswith(ext) for ext in included_extensions)]
            
    for file_name in file_names:
        start = time.time()
        video_name=folder+file_name
        cap = cv2.VideoCapture(video_name)
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
else:
    cap = cv2.VideoCapture(0)



# Rotina de leitura
i=0
image_list=[]
loop_time = time.time()




while(cap.isOpened()):
    # Capture frame-by-frame
    success, webcam_frame = cap.read()

    if success:
        imgContour = webcam_frame.copy()

        # sizeW,sizeH=500,500
        # img = webcam_frame.copy()
        # img=webcam_frame[webcam_frame.shape[0]//2-sizeW//2:webcam_frame.shape[0]//2+sizeW//2,webcam_frame.shape[1]//2-sizeH//2:webcam_frame.shape[1]//2+sizeH//2,:]
        
        if i%fps_limit == 0:
            imgGray = cv2.cvtColor(imgContour,cv2.COLOR_BGR2GRAY)
            ret,imgBin = cv2.threshold(imgGray,20,255,cv2.THRESH_BINARY_INV)
            imgCanny = cv2.Canny(imgBin, 50,50)
            kernel=np.ones((9,9),np.uint8)
            mask=cv2.dilate(imgBin,kernel,iterations=1)
            detections=getContours(mask)
            detect(imgContour,detections)


        print('FPS {} - Análise a cada {} frame(s)'.format(1/(time.time()-loop_time),fps_limit))
        loop_time = time.time()
        i+=1

        cv2.imshow("Frame", cv2.resize(imgContour, (800, 600)))

    # Sai do loop while com a tecla "q"
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
