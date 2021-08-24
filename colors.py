import cv2
import numpy as np
import time

def empty(a):
	pass

inicio=0

cap=cv2.VideoCapture(1)
cap.set(3,640) #width
cap.set(4,480) #height
cap.set(10,100) #brightness
success, img = cap.read()

#img = cv2.imread("SNAPSHOT.png")
#img = cv2.resize(img,(500,500))

#create window with trackbars
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)
cv2.createTrackbar("Hue Min", "Trackbars",0,255,empty)
cv2.createTrackbar("Hue Max", "Trackbars",255,255,empty)
cv2.createTrackbar("Sat Min", "Trackbars",0,255,empty)
cv2.createTrackbar("Sat Max", "Trackbars",255,255,empty)
cv2.createTrackbar("Val Min", "Trackbars",0,255,empty)
cv2.createTrackbar("Val Max", "Trackbars",255,255,empty)

execute=True
while execute:
	success, img = cap.read()
	img = img[:,200:440,:]
	imgHSV = img #cv2.cvtColor(img,cv2.COLOR_BGR2)
	h_min = cv2.getTrackbarPos("Hue Min","Trackbars")
	h_max = cv2.getTrackbarPos("Hue Max","Trackbars")
	s_min = cv2.getTrackbarPos("Sat Min","Trackbars")
	s_max = cv2.getTrackbarPos("Sat Max","Trackbars")
	v_min = cv2.getTrackbarPos("Val Min","Trackbars")
	v_max = cv2.getTrackbarPos("Val Max","Trackbars")
	#print(h_min,h_max,s_min,s_max,v_min,v_max)
	lower = np.array([h_min,s_min,v_min])
	upper = np.array([h_max,s_max,v_max])
	mask = cv2.inRange(imgHSV,lower,upper)
	imgResult = cv2.bitwise_and(img,img,mask=mask)
	
	
	#cv2.imshow("Original", img)
	cv2.imshow("HSV", imgHSV)
	cv2.imshow("Mask", mask)
	cv2.imshow("Result",imgResult)
	k = cv2.waitKey(1) & 0xFF
	
	if k == 27:  # close on ESC key
		execute=False