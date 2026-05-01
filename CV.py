from unittest import result

import cv2
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from flatbuffers.packer import int8
# from keras.src.backend import dtype
from nbformat.sign import algorithms
from scipy.datasets import face
from sympy.physics.units import area
from torchvision.transforms.v2.functional import crop_image

### LEVEL ->>> 1

# read and display an image

img = cv.imread("vishal.jpg")
'''if img is None:
    print("image load nhi hua hai !")
else:
    cv.imshow("my image :",img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
img1 = img.copy()
img1 = cv.resize(img1, (500,300))
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

#cornvert emage different scale
'''pixel = img1[2000,1500]
print("BGR value:",pixel)
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
print("gray value:",gray[2000,1500])
cv.imshow("my image :",gray)
cv.imread("original :",img1)
cv.waitKey(0)
cv.destroyAllWindows()'''

#resie image
'''if img1 is None:
    print("image load nhi hua hai !")
else:
    resized = cv.resize(img1, (2000,1000))
    cv.imshow("my image :",resized)
    cv.waitKey(0)
    cv.destroyAllWindows()'''

# crop a region of interest(ROI)
'''ROI =img1[100:200,100:200]#pixel of image
cv.imshow("my image :",ROI)
cv.waitKey(0)
cv.destroyAllWindows()'''

#save image
'''gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imwrite("vishal_gray.jpg",gray)'''

#flip image
'''flip_H_img = cv.flip(img, 1)#horizontal
flip_V_img = cv.flip(img, 0)#vertical
flip_HV_img = cv.flip(img, -1)#horizantal and vertical both
'''

#rotate image
'''img2 = cv.resize(img1, (1000,900))
h,w = img2.shape[:2]
cx,cy = w//2,h//2
angle = int(input("Enter angle ->>"))
m = cv.getRotationMatrix2D((cx,cy),angle,1)

cos = np.abs(m[0,0])
sin = np.abs(m[0,1])

new_w = int(h*sin+w*cos)
new_h = int(h*cos+w*sin)

m[0,2] += new_w/2- cx
m[1,2] += new_h/2 - cy

rotated = cv.warpAffine(img2,m,(new_w,new_h))
cv.imshow("my image :",rotated)
cv.waitKey(0)
cv.destroyAllWindows()'''

# create a window and drow shape and text
'''n_img = np.full((500,500,3),(0, 0, 0),dtype="uint8")
cv.line(n_img,(0,0),(500,500),(255,255,255),1)
cv.rectangle(n_img,(250,250),(400,100),(255,255,255),1)
cv.line(n_img,(0,100),(500,100),(255,255,255),2)
cv.rectangle(n_img,(350,150),(150,350),(128,128,128),-1)
cv.circle(n_img,(250,250),100,(255,255,255),1)
cv.putText(n_img,"RKvishal",(185,400),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,255),2)
cv.imshow("my image :",n_img)
cv.waitKey(0)
cv.destroyAllWindows()'''

#split and marge color channel
'''b,g,r = cv.split(img1)
cv.imshow("blue image :",b)
cv.imshow("green image :",g)
cv.imshow("red image :",r)
marged = cv.merge((b,g,r))
cv.imshow("marged image :",marged)
cv.waitKey(0)
cv.destroyAllWindows()'''

#BGR to RBG and HCV
'''rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
hcv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
cv.imshow("HCV image :",hcv)
cv.imshow("RGB image :",rgb)
cv.waitKey(0)
cv.destroyAllWindows()'''

# adjust brightness and contrast
'''alpha = float(input("Enter contrast value ->>"))
bita = int(input("Enter brightness value ->>"))
adjust = cv.convertScaleAbs(img1, alpha=alpha, beta=bita)
cv.imshow("adjusted image :",adjust)
cv.waitKey(0)
cv.destroyAllWindows()'''

#bluring
'''aver_b = cv.blur(img1, (11,11))
gaussian_bl = cv.GaussianBlur(img1, (5,5),0)
j = cv.GaussianBlur(img1, (5,5),1)
b = cv.GaussianBlur(img1, (5,5),5)
median_blur = cv.medianBlur(img1, 3)
bilateral_blur = cv.bilateralFilter(img1, 13,55,57)
cv.imshow("averiding_blur image :",aver_b)
cv.imshow("gaussian_blur image :",gaussian_bl)
cv.imshow("median_blur image :",cv.resize(median_blur,None,fx=2,fy=2))#bluring with zooming image
cv.imshow("bilateral_blur image :",bilateral_blur)
cv.waitKey(0)
cv.destroyAllWindows()'''

#detect edges
'''edge1 = cv.Canny(gray,100,200)
edge2 = cv.Canny(img1,100,200)
edge3 = cv.Canny(img1,50,150)
edge4 = cv.Canny(img1,200,300)
cv.imshow("edge1  :",edge1)
cv.imshow("edge2 :",edge2)
cv.imshow("edge3 :",edge3)
cv.imshow("edge4 :",edge4)
cv.waitKey(0)
cv.destroyAllWindows()'''

#apply binary thresholding
'''_, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
V, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
l,thresh2 = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)
k,thresh3 = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO)
n,thresh4 = cv.threshold(gray, 127, 255, cv.THRESH_TRIANGLE)
cv.imshow("thresh image :",thresh)
cv.imshow("thresh1 image :",thresh1)
cv.imshow("thresh2 image :",thresh2)
cv.imshow("thresh3 image :",thresh3)
cv.imshow("thresh4 image :",thresh4)
cv.waitKey(0)
cv.destroyAllWindows()'''

#load and display video
'''vid = cv.VideoCapture("lord_Ram.mp4")
while True:
    ret, frame = vid.read()
    if not ret:
        break
    cv.imshow("frame",frame)
    if cv.waitKey(26) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()'''

#live video capture and grayscale and add FPS on screen
import time
'''cap = cv.VideoCapture(0)
prev_time = 0
while True:
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if not ret:
        break
    cur_time = time.time()
    fbs = 1/(cur_time - prev_time)
    prev_time = cur_time
    cv.putText(frame,f"{int(fbs)}",(20,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv.imshow("video",frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()'''

# extract frame from video
'''count = 0
while True:
    ret,frame = vid.read()

    if not ret:
        break
    if count % 10 == 0:
        cv.imwrite(f"frame_{count}.jpg", frame)
        count += 1
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret1, frame = vid.read()

    cv.imwrite("frame_0_.jpg", frame)

vid.release()
cv.destroyAllWindows()'''

### LEVEL ->>> 2

# implement adapting thresholding
'''adapting = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2 )
cv.imshow("adapting image :",adapting)
cv.waitKey(0)
cv.destroyAllWindows()'''

# erosion and dilation
'''kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(gray,kernel,iterations = 1)
dilation = cv.dilate(gray,kernel,iterations = 1)
cv.imshow("erosion image :",erosion)
cv.imshow("dilation image :",dilation)
cv.waitKey(0)
cv.destroyAllWindows()'''

#opening ,closing and dilation operations
'''_,thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
dilation = cv.morphologyEx(gray,cv.MORPH_DILATE,kernel)
closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
cv.imshow("original image :",thresh)
cv.imshow("opening image :",opening)
cv.imshow("dilation image :",dilation)
cv.imshow("closing image :",closing)
cv.waitKey(0)
cv.destroyAllWindows()'''

#detect contours draw and bounding box
'''_,thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
contour,_ = cv.findContours(thresh1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img1,contour,1,(0,255,0),2)
for  cnt in contour:
    if cv.contourArea(cnt)>500:
        continue

    cv.drawContours(img1,[cnt],-1,(0,0,255),2)

    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)


cv.imshow("edge image :",img1)
cv.waitKey(0)
cv.destroyAllWindows()'''

#count object , find area and perimeter
'''img = cv.imread("rive.png")
img = cv.resize(img,(500,450))
gray2 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# thresh1 = cv.adaptiveThreshold(gray2,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
_,thresh1 = cv.threshold(gray2,127,255,cv.THRESH_TOZERO_INV)
contour,_ = cv.findContours(thresh1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

count = 0
for cnt in contour:
    Area=cv.contourArea(cnt)#area
    perimeters = cv.arcLength(cnt,True)#perimeter

    if Area > 50:
        count += 1

        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # cv.putText(img , f"a : {int(Area)}",(x,y-20),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv.putText(img, f"p : {int(perimeters)}", (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv.putText(img,f"object : {count}",(20,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
cv.imshow("image",img)
cv.imshow("thresh image ",thresh1)
cv.waitKey(0)
cv.destroyAllWindows()'''

#objact cont in live video
'''cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break


    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    _,thresh = cv.threshold(gray_frame,127,255,cv.THRESH_BINARY)
    contour,_ = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    count = 0


    for cnt in contour:
        area = cv.contourArea(cnt)

        if area > 400:
            count += 1

            x,y,w,h = cv.boundingRect(cnt)

            cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv.putText(frame,f"{w}*{h}",(x,y-10),cv.FONT_HERSHEY_PLAIN,0.5,(0,255,0),2)


    cv.putText(frame,f"no. of object : {count}",(20,25),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv.imshow("frame",frame)
    if cv.waitKey(100) & 0xFF == ord('q'):

        break
cap.release()
cv.destroyAllWindows()'''

##circle detect
'''blur = cv.GaussianBlur(gray,(5,5),0)
count =0
circle = cv.HoughCircles(
    blur,
    cv.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=100
)
if circle is not None:
    circles = np.uint16(np.around(circle))


    for (x,y,r) in circles[0]:
        cv.circle(blur,(x,y),r,(0,255,0),2)
        cv.circle(blur,(x,y),2,(0,0,255),2)
        count += 1
cv.putText(blur, f"no. of circle :{count}",(20,30),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
cv.imshow("edge image :",blur)
cv.waitKey(0)
cv.destroyAllWindows()'''

#circle detect in live capture
'''live = cv.VideoCapture(1)

while True:
    ret, frame = live.read()
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur = cv.bilateralFilter(gray_frame,9,50,50)
    circles = cv.HoughCircles(
        blur,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x,y,r) in circles[0]:
            cv.circle(frame,(x,y),r,(255,0,0),2)
            cv.circle(frame,(x,y),2,(255,0,0),3)
            redius = r / 21
            cv.putText(frame,f"redius :{r},{redius}",(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,128,255),2)

    cv.imshow("image :",frame)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break

live.release()
cv.destroyAllWindows()'''

#detect line
'''adge = cv.Canny(img,50,150,apertureSize=3)
lins = cv.HoughLines(adge,1,np.pi/180,150)

if lins is not None:
    for i ,t in lins[:,0]:
        a = np.cos(t)
        b = np.sin(t)

        x0 = a*i
        y0 = b*i

        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))

        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imshow("image",img)
cv.waitKey(0)
cv.destroyAllWindows()'''

# histogram equalization
'''
equalized = cv.equalizeHist(gray)
cv.imshow("original image",gray)
cv.imshow("equalized",equalized)
cv.waitKey(0)
cv.destroyAllWindows()'''

# plot histogram
##1st
'''plt.hist(gray.ravel(),256,[0,256])
plt.title("Histogram")
plt.xlabel("pixel value")
plt.ylabel("number of pixels")

plt.show()'''
##2nd
'''hist = cv.calcHist([gray],[0],None,[256],[0,256])
plt.plot(hist)
plt.title("Histogram")
plt.show()'''
##3rd
'''color = ('b','g','r')
for i ,col in enumerate(color):
    hist = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color = col)
plt.title("color Histogram")
plt.show()'''

# mask a specific color
'''hsv = cv.cvtColor(img1,cv.COLOR_BGR2HSV)
lower = np.array([100,150,0])
upper = np.array([140,255,255])
mask = cv2.inRange(hsv,lower,upper)
result = cv.bitwise_and(img1,img1,mask = mask)
cv.imshow("result",result)
cv.waitKey(0)
cv.destroyAllWindows()'''

# track color object in video
'''cap = cv.VideoCapture(0)
while True :
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    def pick_color(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"position : {x},{y}")
            print(f"HSV vlue : {hsv[x,y]}")
    lower = np.array([100,150,50])
    upper = np.array([140,255,255])

    mask = cv2.inRange(hsv,lower,upper)

    kernel = np.ones((5,5),np.uint8)
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    mask = cv.morphologyEx(mask,cv.MORPH_DILATE,kernel)

    contour,_ = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    if contour:
        cnt = max(contour,key = cv.contourArea)

        if cv.contourArea(cnt) > 500:
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)


            cx = x + w // 2
            cy = y + h // 2

            cv.circle(frame,(cx,cy),5,(0,0,255),-1)
            cv.putText(frame,f"x:{cx} y:{cy}",(x,y-10),cv.FONT_HERSHEY_PLAIN,0.5,(0,255,0),2)

    cv.imshow("frame",frame)
    cv.imshow("mask",mask)
    cv.imshow("hsv",hsv)
    cv.setMouseCallback("mask",pick_color)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()'''

#implement background subtraction
'''live  =cv.VideoCapture(0)
bg_subtrack = cv.createBackgroundSubtractorMOG2() # or (cv.createBackgroundSubtractorKNN())
count = 0

while True:
    ret,frame = live.read()
    if not ret:
        break

    #aplly background subtraction
    fg_mask = bg_subtrack.apply(frame)

    #noise remove (optional but important)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    fg_mask = cv.morphologyEx(fg_mask,cv.MORPH_OPEN,kernel)
    #fg_mask = cv.threshold(fg_mask,200,255,cv.THRESH_BINARY)[1]  better clean detection

    contour,_ = cv.findContours(fg_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        if cv.contourArea(cnt) > 500:
            area = cv.contourArea(cnt)

            if area>500:
                count+=1
                x,y,w,h = cv.boundingRect(cnt)
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        cv.putText(frame,f"detected object : {count}",(30,40),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),2)

    cv.imshow("frame",frame)
    cv.imshow("fg_mask",fg_mask)

    if cv.waitKey(100) & 0xFF == ord('q'):
        break

live.release()
cv.destroyAllWindows()'''

# detect motion btwn frame
#     cap = cv.VideoCapture(0)
#
#     ret, frame1 = cap.read()
#     ret, frame2 = cap.read()
#
#     while cap.isOpened():
#         diff = cv2.absdiff(frame1,frame2)
#
#         gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
#         blur = cv.GaussianBlur(gray,(5,5),0)
#
#         _,thresh = cv.threshold(blur,20,255,cv.THRESH_BINARY)
#
#         deleted = cv.dilate(thresh,None, iterations = 2)
#
#         contour,_ = cv.findContours(deleted,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#
#         for cnt in contour:
#             if cv.contourArea(cnt) > 500:
#                 x,y,w,h = cv.boundingRect(cnt)
#                 cv.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)
#
#             frame1 = frame2
#             cv.imshow("frame1",frame1)
#             if cv.waitKey(100) & 0xFF == ord('q'):
#                 break
#
#     cap.release()
#     cv.destroyAllWindows()


# object tracker
'''vid = cv.VideoCapture(0)
mg = cv.createBackgroundSubtractorMOG2()
objects = {}
object_id = 0
while True:
    ret,frame = vid.read()
    if not ret:
        break



    fg_mask = mg.apply(frame)
    _, fg_mask = cv.threshold(fg_mask,200,255,cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    fg_mask = cv.morphologyEx(fg_mask,cv.MORPH_OPEN,kernel)

    contour,_ = cv.findContours(fg_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    new_objects = {}
    for cnt in contour:
        if cv.contourArea(cnt) > 500:
            area = cv.contourArea(cnt)
            if area>500:
                x,y,w,h = cv.boundingRect(cnt)

                cx,cy = int((x+w/2)/2),int((y+h/2)/2)
                matched = False
                for i ,(px,py) in objects.items():
                    distance = np.hypot(cx-px,cy-py)

                    if distance < 50:
                        new_objects[i] = (cx,cy)
                        matched = True
                        break

                if not matched:
                    new_objects[object_id] = (cx,cy)
                    object_id+=1

                cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv.circle(fg_mask,(cx,cy),5,(0,0,255),-1)

            objects = new_objects

            for i,(cx,cy) in objects.items():
                cv.putText(frame,f"id {i}",(cx,cy),cv.FONT_HERSHEY_PLAIN,0.7,(0,0(),0),1)

            cv.imshow("frame",frame)

            if cv.waitKey(100) & 0xFF == ord('q'):
                break

vid.release()
cv.destroyAllWindows()'''


#perform perspective transformation
'''vaid = cv.VideoCapture(0)

while True:
    ret,frame = vaid.read()


    if not ret:
        break

    pts1 = np.float32([
        [50,500],
        [200,200],
        [50,200],
        [200,200],
    ])

    pts2 = np.float32([
        [0,0],
        [300,0],
        [0,300],
        [300,300],
    ])

    metrix = cv.getPerspectiveTransform(pts1,pts2)

    result = cv.warpPerspective(frame,metrix,(640,480))

    def mouse_click(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"position : ({x},{y})")


    cv.imshow("result",result)
    cv.imshow("frame",frame)
    cv.setMouseCallback("result",mouse_click)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break


print(frame.shape)
vaid.release()
cv.destroyAllWindows()'''

# Edge detection
'''vid = cv.VideoCapture(0)
count = 0
while True:
    ret,frame = vid.read()
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    if not ret:
        break

    dst = cv.cornerHarris(frame_gray,2,3,0.04)
    for d in dst:
        count += 1
        
        cv.putText(frame,f"corners:{count}",(23,30),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
    frame[dst>0.01*dst.max()] = [0,0,255]


    cv.imshow("result",frame)
    if cv.waitKey(50) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()'''

# detect corner using Shi _Tomasi
'''vid = cv.VideoCapture(0)
count = 0

while True:
    ret,frame = vid.read()
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    if not ret:
        break

    corners = cv.goodFeaturesToTrack(frame_gray,50,0.05,50)

    corners = np.int32(corners)

    for corner in corners:
        x,y = corner.ravel()
        cv.circle(frame,(x,y),5,(0,0,255),-1)
        count += 1
        cv.putText(frame ,f" corners : {corners}",(20,30),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
    cv.imshow("result",frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
print(f" numbers of corner : {count}")
vid.release()
cv.destroyAllWindows()
## for  image 
corners = cv.goodFeaturesToTrack(gray,50,0.01,10)

corners = np.int32(corners)

for corner in corners:
    x,y = corner.ravel()
    cv.circle(img1,(x,y),5,(0,0,255),-1)
    cv.imshow("result",img1)
cv.waitKey(0)
cv.destroyAllWindows()'''

#slicing in image
tem = gray[100:300,200:400]
'''cv.imshow("result",tem)
cv.imshow("frame",gray)
cv.waitKey(0)
cv.destroyAllWindows()'''

#template matching
''''''
'''matched = cv.matchTemplate(gray,tem,cv.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(matched)

h,w = tem.shape
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv.rectangle(gray,top_left,bottom_right,(0,255,0),2)
cv.imshow("result",gray)
cv.imshow("frame",matched)
cv.waitKey(0)
cv.destroyAllWindows()
#for video or live
cap = cv.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    if not ret:
        break

    result = cv.matchTemplate(frame_gray,ret_gray,cv.TM_CCOEFF_NORMED)
    min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)

    h,w = frame_gray.shape

    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(frame,top_left,bottom_right,(0,255,0),2)

    cv.imshow("result",frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()'''

#blend two image together
'''tem = cv.resize(tem,(gray.shape[1],gray.shape[0]))
blend = cv.addWeighted(gray,0.7,tem,0.3,0)
cv.imshow("result",blend)
cv.waitKey(0)
cv.destroyAllWindows()'''

#create an image
'''c_img = np.ones((500,500,3),dtype= "uint8")
cv.line(c_img,(250,100),(150,300),(0,0,255),2)
cv.line(c_img,(150,300),(350,300),(0,0,255),2)
cv.line(c_img,(350,300),(250,100),(0,0,255),2)

y,x1,x2 = 280,160,340
for i in range(10):
    cv.line(c_img,(x1,y),(x2,y),(0,255,0),2)

    y -= 20
    x1 += 10
    x2 -= 10
def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print("position:",x,y)

cv.imshow("result",c_img)
cv.setMouseCallback("result",mouse_click)
cv.waitKey(0)
cv.destroyAllWindows()'''

#image pyramid
'''layer = img1.copy()
gp = [layer ]

for i in range(10):
    layer = cv.pyrDown(layer)
    gp.append(layer)
    cv.imshow(f"level {i+1}",layer)
cv.waitKey(0)
cv.destroyAllWindows()'''

# face detecting
'''face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades+"haarcascade_frontalface_default.xml"
)

eyes_cascade = cv.CascadeClassifier(
    cv.data.haarcascades+"haarcascade_eye.xml"
)
#
# faces = face_cascade.detectMultiScale(
#     gray,
#     scaleFactor=1.3,
#     minNeighbors=5,
# )
# for (x,y,w,h) in faces:
#     cv.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
#
# # cv.imshow("result",faces)
# cv.imshow("Faces",img1)
# cv.waitKey(0)
# cv.destroyAllWindows()

cap = cv.VideoCapture(0)

while True:
    ret,frame = cap.read()
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray_frame,1.3,5)
    for (x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        roi_gray = gray_frame[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

    eyes = eyes_cascade.detectMultiScale(roi_gray,1.2,10)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)


    cv.imshow("result",frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()'''

# face blurring in video
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades+"haarcascade_frontalface_default.xml"
)

''''cap2 = cv.VideoCapture(0)

while True:
    ret,frame = cap2.read()

    if not ret:
        print("camera error!")
        break
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray_frame,1.3,5)

    for (x,y,w,h) in face:
        face = frame[y:y+h,x:x+w]

        #blur
        face = cv.GaussianBlur(face,(99,99),33)

        frame[y:y+h,x:x+w] = face

    cv.imshow("result",frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap2.release()
cv.destroyAllWindows()'''''

#extract ROI (Region of Interest) from detected part
'''face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades+"haarcascade_frontalface_default.xml"
)
vid = cv.VideoCapture(0)
while True:
    ret,frame = vid.read()
    if not ret:
        print("camera error!")
        break

    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray_frame,1.3,5)
    for (x,y,w,h) in face:
        face = frame[y:y+h,x:x+w]#ROI
        cv.imshow("result",face)
    cv.imshow("Faces",frame)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()'''

# simple document scaner
'''
sca = cv.VideoCapture(1)

while True:
    ret,frame = sca.read()
    if not ret:
        print("camera error!")
        break
    frame = cv.resize(frame,(640,480))
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray_frame,(5,5),0)
    edge = cv.Canny(blur,30,100)
    contour,_ = cv.findContours(edge,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    if len(contour) != 0:
        cnt = max(contour,key=cv.contourArea)
        if cv.contourArea(cnt) > 10000:
            peri = cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            # pts = approx.reshape(4,2)
            # scan = cv.adaptiveThreshold(
            #     frame,255,
            #     cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            #     cv.THRESH_BINARY,
            #     11,2
            # )
            cv.drawContours(frame,[approx],0,(0,255,0),3)
    cv.imshow("result",frame)
    cv.imshow("edge",edge)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
sca.release()
cv.destroyAllWindows()'''


### level -->>> 3

#ORB feature detection
 #with imag
'''orb = cv.ORB_create()

keypoints, descriptors = orb.detectAndCompute(img1,None)
print(f"keypoints: {keypoints}, \n descriptors: {descriptors}")
img_c = cv.drawKeypoints(img1,keypoints,None,color=(0,255,0))

cv.imshow("img1",img_c)
cv.waitKey(0)
cv.destroyAllWindows()
 #with video
cap = cv.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if not ret:
        print("camera error!")
        break

    keypoints1, descriptors1 = orb.detectAndCompute(frame,None)
    frame_c = cv.drawKeypoints(frame,keypoints1,None,color=(255,0,0))
    cv.imshow("img1",frame_c)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()'''

# features matching in two image
'''orb = cv.ORB_create()
cap = cv.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if not ret:
        print("camera error!")
        break
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    keypoints1, descriptors1 = orb.detectAndCompute(frame,None)
    keypoint, descriptor = orb.detectAndCompute(img1,None)

    bf = cv.BFMatcher(cv.NORM_HAMMING,True)
    matched = bf.match(descriptor,descriptors1)
    matched = sorted(matched,key = lambda x:x.distance)
    result = cv.drawMatches(frame,keypoints1,img1,keypoint,matched[:30],None)
    cv.imshow("img1",result)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()'''

# panorama using image
'''orb = cv.ORB_create()
k1, d1 = orb.detectAndCompute(img1,None)
k2, d2 = orb.detectAndCompute(img,None)
bf = cv.BFMatcher(cv.NORM_HAMMING)
matches = bf.knnMatch(d1,d2,k=2)
good =[]
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
 #points extract
pnt1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
pnt2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

H,_ = cv2.findHomography(pnt2,pnt1,cv2.RANSAC,5.0)# homography
result = cv.warpPerspective(img,H,(img1.shape[1]+img.shape[1],img1.shape[0]))#warp
result[0:img1.shape[0],0:img1.shape[1]] = img1 #stitch
cv.imshow("result",result)
cv.waitKey(0)
cv.destroyAllWindows()'''

# object track using optical flow
'''vid = cv.VideoCapture(0)

 # 1st frame
ret,old_frame = vid.read()
old_gray = cv.cvtColor(old_frame,cv.COLOR_BGR2GRAY)

 #good feature (point to track)
p0 = cv.goodFeaturesToTrack(old_gray,100,0.3,7)

 #mask for drawing
mask = np.zeros_like(old_frame)
while True:
    ret,frame = vid.read()
    gray1 = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    if not ret:
        print("camera error!")
        break
    #optical flow
    p1,st,err = cv.calcOpticalFlowPyrLK(old_gray,gray1,p0,None)
    if p1 is None or st is None:
        cv.putText(frame,"no point for track",(10,10),2,1,(0,0,255),3)
        cv.imshow("frame",frame)

        old_gray = gray1.copy()
        p0 = cv.goodFeaturesToTrack(old_gray,100,0.3,7)
        continue
    #salect good point
    good_new = p1[st==1]
    good_old = p0[st==1]

    #draw tracking
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        cv.line(mask, (int(a),int(b)),(int(c),int(d)),(0,255,0),2)
        cv.circle(frame,(int(a),int(b)),5,(0,0,255),-1)

    im = cv.add(frame,mask)

    cv.imshow("frame",im)

    old_gray = gray1.copy()
    p0 = good_new.reshape(-1,1,2)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()'''

# implement lucas-Kanade optical flow
'''cap = cv.VideoCapture(0)
 #1st frame
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame,cv.COLOR_BGR2GRAY)

 #feature to track (shi-Tomasi)
p0 = cv.goodFeaturesToTrack(
    old_gray,
    maxCorners = 100,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 5,
)

#mask drawing
mask = np.zeros_like(old_frame)
while True:
    ret, frame = cap.read()

    if not ret:
        print("camera error!")
        break
    gray1 = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    #lucas-kanade
    p1, st, err = cv.calcOpticalFlowPyrLK(
        old_gray,gray1,p0,None,
        winSize = (15,15),
        maxLevel = 5,
        criteria = (cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    if p0 is None or st is None:
        cv.putText(frame,"no point for track",(10,10),2,1,(0,0,255),3)
        cv.imshow("frame",frame)

        old_gray = gray1.copy()
        p0 = cv.goodFeaturesToTrack(old_gray,100,0.3,7)
        continue
    # valid points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw tracking
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        cv.line(mask, (int(a),int(b)),(int(c),int(d)),(0,255,0),2)
        cv.circle(frame,(int(a),int(b)),5,(0,0,255),-1)

    img = cv.add(frame,mask)
    cv.imshow("frame",img)

    #update frame
    old_gray = gray1.copy()
    p0 = good_new.reshape(-1,1,2)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()'''


#SIFT(Scale Invariant Feature Transform) keypoint detect
'''cp = cv.VideoCapture(0)
while True:
    ret,frame =cp.read()
    if not ret:
        print("camera error!")
        break

    gray1 = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    #SIFT
    sift = cv.SIFT_create()

    #detect+compute
    kp1, des1 = sift.detectAndCompute(gray1,None)

    #draw keypoints
    img_kp = cv.drawKeypoints(frame,kp1,None)
    bf = cv.BFMatcher()


    cv.imshow("frame",img_kp)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
cp.release()
cv.destroyAllWindows()'''

#SIFT points matching
'''vid = cv.VideoCapture(0)


while True:
    ret,old_frame = vid.read()
    old_gray = cv.cvtColor(old_frame,cv.COLOR_BGR2GRAY)
    ret, frame = vid.read()
    if not ret:
        print("camera error!")
        break
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    k1, des1 = sift.detectAndCompute(old_frame,None)
    k2, des2 = sift.detectAndCompute(frame,None)

    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)


    result = cv.drawMatches(old_frame,k1,frame,k2,good,None)
    cv.imshow("frame",result)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()'''

# FLANN(Fast Library for Approximate Nearest neighbors )-based matching
pas = cv.imread("passportt.png")
'''sift = cv.SIFT_create()
k1 ,d1 = sift.detectAndCompute(img1,None)
k2, d2 = sift.detectAndCompute(pas,None)

index_param = dict(algorithm=1, trees=5)
search_param = dict(checks=5)
Fl = cv.FlannBasedMatcher(index_param,search_param)
match = Fl.knnMatch(d1,d2,k=2)

good = []
for m,n in match:
    if m.distance < 0.75*n.distance:
        good.append(m)

result = cv.drawMatches(img1,k1,pas,k2,good,None)
result = cv.resize(result,(1500,900))
cv.imshow("frame",result)
cv.waitKey(0)
cv.destroyAllWindows()'''

# homography estimation
'''orb = cv.ORB_create()
p1, d1 = orb.detectAndCompute(img1,None)
p2, d2 = orb.detectAndCompute(pas,None)
bf = cv.BFMatcher(cv.NORM_L2)
match = bf.knnMatch(d1,d2,k=2)
good = []
for m,n in match:
    if m.distance < 0.75*n.distance:
        good.append(m)
if len(good)>6:
    pnt1 = np.float32([p1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pnt2 = np.float32([p2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
     #Homography
    h,_ =cv.findHomography(pnt1,pnt2,cv.RANSAC,5.0)
     #warp
    result = cv.warpPerspective(img1,h,(img1.shape[1],img1.shape[0]))
    cv.imshow("frame",result)
    cv.waitKey(0)
    cv.destroyAllWindows()
else :
    print("no matches")'''

#detect and match logo
'''log = cv.resize((cv.imread("apple.png")),(500,300))
lower = np.array([0,0,0])
upper = np.array([50,50,50])
logo =cv.inRange(log,lower,upper)
log[logo != 0] = [255,255,255]
cv.imshow("apple",logo)
cv.waitKey(0)
cv.destroyAllWindows()
img2 = cv.resize((cv.cvtColor((cv.imread("img.png")),cv.COLOR_BGR2GRAY)),(500,300))
cv.imshow("logo",img2)

sift = cv.SIFT_create()

k1,d1 = sift.detectAndCompute(logo,None)
k2, d2 = sift.detectAndCompute(img2,None)
bf = cv.BFMatcher(cv.NORM_L2)
matched = bf.knnMatch(d1,d2,k=2)
good =[]
for m,n in matched:
    if m.distance < 0.80*n.distance:
        good.append(m)
print(len(good))
if len(good):
    p1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    p2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    m,mask = cv.findHomography(p1,p2,cv.RANSAC,5.0)

    h,w = logo.shape
    pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)

    dst = cv.perspectiveTransform(pts,m)

    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3)
    cv.imshow("frame",img2)
    cv.waitKey(0)
    cv.destroyAllWindows()'''

# face recognition system

'''
vid = cv.VideoCapture(0)
while True:
    ret,frame = vid.read()
    if not ret:
        print("camera not opened! ")
        break

    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray_frame,1.3,2)
    for (x,y,w,h) in face:
        face = gray_frame[y:y+h,x:x+w]

        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.train(face,np.array([]))
        label, confidence = recognizer.predict(face)
        if confidence<60:
            name = "Vishal"
        else:
            name = "Unknow"

        cv.imshow("face",face)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
vid.release()
cv.destroyAllWindows()'''

# object detect with DNN
'''net = cv.dnn.readNetFromTensorflow(
    "frozen_inference_graph.pb",
    "ssd_mobilenet_v3_large_coco_2020_01_28.pbtxt"
)

classes = []
with open("coco.names","r") as f:
    classes = f.read().strip().split("\n")

h,w = img1.shape[:2]

blob = cv.dnn.blobFromImage(img1,size=(300,320),swapRB=True,crop=False)
net.setInput(blob)
outputs = net.forward()

for d in outputs[0,0,:,:]:
    confidence = d[2]

    if confidence>0.5:

        class_id = int(d[3])
        x1 = int(d[3]*w)
        y1 = int(d[4]*h)
        x2 = int(d[5]*w)
        y2 = int(d[6]*h)
        label = classes[class_id]

        cv.rectangle(img1,(x1,y1),(x2,y2),(0,255,0),2)
        cv.putText(img1,label,(x1,y1-10),
                   cv.FONT_HERSHAY_SIMPLEX,0,255,0)

cv2.imshow("frame",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#detect number plate
import pytesseract
import re
import csv
import datetime
'''vid = cv.VideoCapture("car.mp4")
while True:
    ret,frame = vid.read()
    if not ret:
        print("camera not opened! ")
        break

    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur_frame = cv.GaussianBlur(gray_frame,(5,5),0)

    edge = cv.Canny(blur_frame,100,200)

    contours,_ = cv.findContours(edge,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    plate = None
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)

        ratio = w/h

        if 2< ratio < 5 and w>60 and h>30:
            roi = gray_frame[y:y+h,x:x+w]
            roi_edge = cv.Canny(roi,100,200)

            e_p = cv.countNonZero(roi_edge)
            density = e_p/(w*h)
            if density>0.222:
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            if plate is not None and plate.size >0:
                text = pytesseract.image_to_string(plate)

                text = text.strip()
                text = re.sub(r'[^A-Z0_9]','',text)
                with open("plate.csv","a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([text,datetime.datetime.now()])

    cv.imshow("frame",frame)
    # cv.imshow("edge",edge)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()'''

#hand gesture
import mediapipe as mp

'''mp_hand = mp.solutions.hands
hands = mp_hand.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

while True:
    success,frame = cap.read()
    frame_rbg = cv2.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = hands.process(frame_rbg)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,handLms,mp_hand.HAND_CONNECTIONS)

        tips= [4,8,12,16,20]
        fingers = []

        if handLms.landmark[4].x< handLms.landmark[3].x :
            fingers.append(1)
        else:
            fingers.append(0)

        for tip in tips[1:]:
            if handLms.landmark[tip].y < handLms.landmark[tip-2].y :
                fingers.append(1)
            else:
                fingers.append(0)

        count= fingers.count(1)
        cv.putText(frame,f"Fingers ->> {count}",(10,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


    cv.imshow("mask",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()'''

#segment foreground using GrabCut
'''vid = cv.VideoCapture(0)
while True:
    ret , frame = vid.read()
    if not ret:
        print("camera not opened! ")
        break
    mask = np.zeros(frame.shape[:2],dtype="uint8")

    bgdModel = np.zeros((1, 65),np.float64)
    fgdModel = np.zeros((1, 65),np.float64)

    rect = (50,50,300,200)

    cv.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),255,255).astype('uint8')
    contour ,_ = cv.findContours(mask2,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    results = mask*mask2[:,:np.newaxis]

    cv.imshow("result",results)
    cv.imshow("frame",frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()'''


# photo = cv.imread(input("drop your photo:"))
# photo = cv.resize(photo,(800,500))
# cv.imshow("photo",photo)
# cv.waitKey(0)
# cv.destroyAllWindows()

# watershed algorithm for segmentation
'''img1 = cv.resize(img1,(600,400))
gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

_,thrsh = cv.threshold(gray1,0,255,cv.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(thrsh,cv.MORPH_OPEN,kernel, iterations = 2)

sure_bg = cv.dilate(opening,kernel,iterations = 3)

dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)

_, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

unknown = cv.subtract(sure_bg,sure_fg)

_,marker = cv.connectedComponents(sure_fg)

marker = marker+1
marker[unknown==255] = 0
marker = cv.watershed(img1,marker)

img1[marker==-1]=[255,0,0]

cv.imshow("img1",img1)
cv.waitKey(0)
cv.destroyAllWindows()'''

#track multi object simultaneously
'''from scipy.spatial import distance as dist

vid = cv.VideoCapture(1)

object_id = 0

objects = {}

def get_centroid(x,y,w,h):
    return (int(x+w/2),int(y+h/2))
while True:
    ret,frame = vid.read()
    if not ret:
        print("camera not opened! ")
        break

    gray1 = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    _,thrsh = cv.threshold(gray1,127,255,cv.THRESH_BINARY)
    contour,_ = cv.findContours(thrsh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    new_contour = []
    for cnt in contour:
        if cv.contourArea(cnt) > 1000:
            x,y,w,h = cv.boundingRect(cnt)
            c = get_centroid(x,y,w,h)
            new_contour.append(c)

            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    new_objects= {}
    for c in new_contour:
        matched = False

        for object_id ,old_c in objects.items():
            distance = dist.euclidean(c,old_c)

            if distance < 100:
                new_objects[object_id] = old_c
                matched = True
                break
        if not matched:
                new_objects[object_id] = c
                object_id +=1

    objects = new_objects

    for object_id,c in objects.items():
        cv.putText(frame,f"id {object_id}",c,cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv.imshow("mask",frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()'''

# detect anomalies in video
'''cap = cv.VideoCapture(0)
 #background subtractor
fgbg = cv.createBackgroundSubtractorMOG2(history=500,varThreshold=50)

while True:
    ret,frame = cap.read()
    if not ret:
        print("camera not opened! ")
        break
    #resize for stability
    frame = cv.resize(frame,(600,400))
    
    #1. foreground mask
    fgmask = fgbg.apply(frame)
    
    #2. noice remove 
    kernel =cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    fgmask = cv.morphologyEx(fgmask,cv.MORPH_OPEN,kernel,iterations=2)
    
    #3.contours
    contour,_ = cv.findContours(fgmask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    anomaly_detected = False
    for cnt in contour:
        area = cv.contourArea(cnt)
        
        #4. filter(important)
        if area > 2000:
            x,y,w,h = cv.boundingRect(cnt)

            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            anomaly_detected = True

    #5. show anomaly
    if anomaly_detected:
        cv.putText(frame,"ANOMALY detected ",(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv.imshow("frame",frame)
    cv.imshow("fgmask",fgmask)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()'''

#create heatmap
'''cap = cv.VideoCapture(0)
heatmap = None

fgbg = cv.createBackgroundSubtractorMOG2()

while True:
    ret,frame = cap.read()
    if not ret:
        print("camera not opened! ")
        break

    frame = cv.resize(frame,(600,400))
    #1.foreground (motion)
    fgmask = fgbg.apply(frame)

    #2.clean noise
    kernel =  np.ones((3,3),np.uint8)
    fgmask = cv.morphologyEx(fgmask,cv.MORPH_OPEN,kernel)
    #3.for show detected part
    contour, _ = cv.findContours(fgmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    anomaly_detected = False
    for cnt in contour:
        area = cv.contourArea(cnt)

        # 4. filter(important)
        if area > 2000:
            x, y, w, h = cv.boundingRect(cnt)

            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #5.init heatmap
    if heatmap is None:
        heatmap = np.zeros_like(fgmask, dtype=np.float32)
    
    #6.accumulate motion 
    heatmap += fgmask
    
    #6.normalize
    heatmap_norm =(cv.normalize(heatmap,heatmap,0,255,cv.NORM_MINMAX)).astype(np.uint8)
    
    #7.apply color map 
    heatmap_color = cv.applyColorMap(heatmap_norm,cv.COLORMAP_JET)
    
    #8.overlay
    overlay = cv.addWeighted(frame,1,heatmap_color,1,0)

    cv.imshow("frame",overlay)
    cv.imshow("heatmap",heatmap)
    cv.imshow("fgmask",frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()'''

#people counter system
'''
 
 ##in image
people = cv.imread("people.jpg")
hog= cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
boxes,_ = hog.detectMultiScale(people,winStride=(8,8))
count = 0

for (x,y,w,h) in boxes:
    count += 1
    cv.rectangle(people,(x,y),(x+w,y+h),(0,255,0),2)
cv.putText(people,f"people:{count}",(30,20),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
cv.imshow("people",people)
cv.waitKey(0)
cv.destroyAllWindows()

 # in video or live
vid = cv.VideoCapture("people_v.mp4")

fgbg = cv.createBackgroundSubtractorMOG2()
count = 0
line_y = 250
while True:
    ret,frame = vid.read()
    if not ret:
        print("video not opened! ")
        break

    frame = cv.resize(frame,(600,400))
    fgmask = fgbg.apply(frame)

    #clean noise
    kernel =  np.ones((3,3),np.uint8)
    fgmask = cv.morphologyEx(fgmask,cv.MORPH_OPEN,kernel)

    contour,_ = cv.findContours(fgmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        area = cv.contourArea(cnt)

        if area > 2000:
            x,y,w,h = cv.boundingRect(cnt)
            cx,cy = x+w//2,y+h//2

            #draw box
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv.circle(frame,(cx,cy),5,(0,0,255),2)

            #crossing logic
            if line_y -5<cy<line_y+5:
                count+=1
    #draw line
    cv.line(frame,(0,line_y),(640,line_y),(0,0,255),2)
    #display count of people
    cv.putText(frame,f"peoples:{count}",(30,20),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv.imshow("frame",frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()'''

#detect lane lines in road
'''road = cv.resize((cv.imread("road.jpg")),(640,480))
gray_road = cv.cvtColor(road,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray_road,(5,5),0)
edge = cv.Canny(blur,50,150)
mask = np.zeros_like(edge)

height, width = gray_road.shape
 #traingular region 
roi = np.array([[
    (0,height),
    (width//2,height//2),
    (width,height),
]],dtype=np.int32)

cv.fillPoly(mask,roi,(255,255,255))

roi_edges = cv.bitwise_and(edge,mask)
 #Hough lines 
line = cv.HoughLinesP(roi_edges,1,np.pi/180,50,minLineLength=50,maxLineGap=100)
 #draw line 
line_img = np.zeros_like(road)
if line is not None:
    for l in line:
        x1,y1,x2,y2 = l[0]
        cv.line(line_img,(x1,y1),(x2,y2),(0,255,0),2)
#voerlay
result = cv.addWeighted(road,0.5,line_img,0.5,0)
cv.imshow("result",result)
cv.waitKey(0)
cv.destroyAllWindows()'''

#estimate depth
import torch

'''model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/miDaS",model_type)
midas.eval()

transforms = torch.hub.load("intel-isl/miDaS","transforms")
transform = transforms.small_transform

im = cv.imread("road.jpg")
img_rgb = cv.cvtColor(im,cv.COLOR_BGR2RGB)

input_batch = transform(img_rgb)

with torch.no_grad():
    prediction = midas(input_batch)

prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(0),
    size=im.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()

depth_map = prediction.cpu().detach().numpy()

depth_map = cv.normalize(depth_map,None,0,255,cv.NORM_MINMAX)
depth_map = depth_map.astype(np.uint8)

cv.imshow("depth map ",depth_map)
cv.waitKey(0)
cv.destroyAllWindows()'''

#calibrate image
'''import glob
chessboard_size = (9,6)

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoint = []
imgpoint = []

image = glob.glob("road.jpg")

for frame in image:
    img = cv.imread(frame)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray,chessboard_size,None)

    if ret:
        objpoint.append(objp)
        imgpoint.append(corners)

        cv.drawChessboardCorners(img,chessboard_size,corners,ret)
        cv.imshow("img",img)
        cv.waitKey(200)
cv.destroyAllWindows()

ret,mtx,dist,rvecs,tvecs =cv2.calibrateCamera(
    objpoint,imgpoint,gray.shape[::-1],None,None
)

print("mtx:",mtx)
print("dist:",dist)

img= cv.imread(image[0])
h,w = img.shape[:2]

newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

dst = cv.undistort(img,mtx,dist,None,newcameramtx)

cv.imshow("dst",dst)
cv.waitKey(0)
cv.destroyAllWindows()'''


### LEVEL -->>> 4

# traffic monitoring system
from  ultralytics import YOLO
'''model = YOLO("yolov8n.pt")
vid = cv.VideoCapture("traffic.mp4")
line_y = 300
count = 0
tracked_ids = set()
while True:
    ret,frame = vid.read()
    if not ret:
        print("video error")
        break

    frame = cv.resize(frame,(640,480))

    result = model.track(frame,persist=True)

    if result[0] is not None:
        boxes = result[0].boxes.xyxy.cpu().numpy()
        ids = result[0].boxes.id.cpu().numpy()

        for box , object_id in zip(boxes,ids):
            x1,y1,x2,y2 = map(int,box)

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            cv.rectangle(frame,(x2,y2),(x1,y1),(0,255,0),2)
            cv.putText(frame,f"ID {int(object_id)}",(x1,y1-10),cv.FONT_HERSHEY_TRIPLEX,0,255,0,2)

            if cy > line_y and object_id not in tracked_ids:
                count+= 1
                tracked_ids.add(object_id)

    cv.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    cv.putText(frame,f"count: {count}",(20,30),cv.FONT_HERSHEY_TRIPLEX,0,255,3)

    cv.imshow("frame",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()'''

# track vehicle and count
'''model = YOLO("yolov8n.pt")
line_y = 300
count = 0
vid = cv.VideoCapture("traffic.mp4")
while True:
    ret,frame = vid.read()
    if not ret:
        print("video error")
        break

    frame = cv.resize(frame,(640,480))
    result = model.track(frame,persist=True)
    if result[0] is not None:
        boxes = result[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1,y1,x2,y2 = map(int,box)
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            cv.rectangle(frame,(x2,y2),(x1,y1),(0,255,0),2)

            count +=1

    cv.putText(frame,f"total vehicle: {count}",(20,30),cv.FONT_HERSHEY_SIMPLEX,0,255,2)
    cv.line(frame,(0,line_y),(frame.shape[1],line_y),(0,0,255),2)
    cv.imshow("frame",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()'''

# Smart parking System
'''cap = cv.VideoCapture("parking.mp4")
  #define slot
slot = [
    (50,100,80,150),
    (150,100,80,150),
    (250,100,80,150)
]

while True:
    ret,frame = cap.read()
    if not ret:
        print("video error")
        break

    frame = cv.resize(frame,(640,480))
    #for clear and stabile
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray_frame,(5,5),0)

    #area track
    thresh = cv.adaptiveThreshold(
        blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,25,16
    )

    free_slot = 0
    for (x,y,w,h) in slot:
        roi = thresh[y:y+h,x:x+w]

        count = cv.countNonZero(roi)

        #threshold tune
        if count < 2000:
            color = (0,0,255)# free space
            free_slot += 1
        else:
            color = (0,0,0)

        cv.rectangle(frame,(x,y),(x+w,y+h),color,2)
    cv.putText(frame,f"free space: {free_slot}",(20,30),cv.FONT_HERSHEY_SIMPLEX,0,255,2)
    cv.imshow("frame",frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()'''

#