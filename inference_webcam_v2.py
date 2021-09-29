from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os
import time

classes=['black','metal','red']
model_name = "MobileNetV2"
model=load_model(model_name+'/model.h5')

modo = int(input("0 Para utilizar video\n1 Para utiliza Webcam\n2 Para uma imagem\n"))

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
elif modo==1:
    cap = cv2.VideoCapture(1)
else:
    img_name=input("Qual imagem?")
    img=cv2.imread(img_name)
    img_ = image.array_to_img(img)
    # img = region
    img_ = img_.resize((224, 224))

    x_=image.img_to_array(img_)
    x_=x_/255
    x_=np.expand_dims(x_,axis=0)

    y_pred=model.predict(x_)
    
    resultado = int(np.argmax(y_pred))
    print("Classe: " + classes[resultado])

if modo != 2:
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
                img_ = image.array_to_img(webcam_frame)
                # img = region
                img_ = img_.resize((224, 224))

                x_=image.img_to_array(img_)
                x_=x_/255
                x_=np.expand_dims(x_,axis=0)

                y_pred=model.predict(x_)
                resultado = int(np.argmax(y_pred))

                print('FPS {} - Análise a cada {} frame(s) - classe {}'.format(1/(time.time()-loop_time),fps_limit,classes[resultado]))
            loop_time = time.time()
            i+=1

            cv2.imshow("Frame", cv2.resize(imgContour, (800, 600)))

        # Sai do loop while com a tecla "q"
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
