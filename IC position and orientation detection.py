#libraries 

import cv2
import numpy as np


#functions

def left_contour(contours):
#This function checks which contour is the left most one
#It returns that contour and the original set of contours with the left most one removed
    
    X=[]
    for i in range (len(contours)):
        rect=cv2.minAreaRect(contours[i])
        X.append(rect[0][0])

        if i==0:
            left=contours[i]
        elif X[i]<X[i-1]:
            left=contours[i]
    contours.remove(left)        
    return left,contours


def compute_ratio(contour,r_metric):
#This fuction takes a circle shaped reference contour and its radius in mm and returns a ratio (mm:pixels) to compute any dimention on the image
    area=cv2.contourArea(contour)
    r_pixel = area / (np.power(np.pi, 2))
    ratio = r_metric / r_pixel
    return ratio


def detect_ic_position_angle(img):
#This function is used to detect the ics in a frame and return their exact position from the camera and their rotational angle

#This function takes the frame as an argument and returns the image with the contours drawn, the (x,y) coordinate of the center of each contour and the angle at which they are rotated

    #code starts here
    coordinates,angles=[],[]    
    gray_frame=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                                             #transforming the frame to grayscale for easier computation and handling                     
    y,x=gray_frame.shape                                                                                        #getting the size of the frame to show the center cross
    recoloured_frame=cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)
    cv2.line(recoloured_frame,((x/2),0),((x/2),y),(0,255,255),2)                                                #drawing center lines on the frame
    cv2.line(recoloured_frame,(0,(y/2)),(x,(y/2)),(0,255,255),2)
    
    kernel = np.ones((29,29),np.uint8)                                                                          #creating a kernel for the closing process
    
    frame_closing=cv2.morphologyEx(gray_frame,cv2.MORPH_CLOSE,kernel)                                           #applying morphological closing to the image to remove any defects
                                                                                                                #caused by the image capturing process
    
    median_blured=cv2.medianBlur(frame_closing,11)                                                              #applying a median filter to remove any salt and pepper noise
    
    diff_img=255- cv2.absdiff(gray_frame,median_blured)                                                         #trying to remove any shadows that exist in the image

    norm_img=gray_frame.copy()
    
    norm_img=cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)      #normalizing the image(increasing the contrast)

    norm_edges=cv2.Canny(diff_img,100,200)                                                                      #now finding the edges in the image to find the contours
    norm_edges_closing=cv2.morphologyEx(norm_edges,cv2.MORPH_CLOSE,kernel)                                      #applying morpholigical closing to fill in the detected shapes                             
    norm_edges_closing=cv2.medianBlur(norm_edges_closing,15)                                                    #applying a median filter to exclude any noise

    image,contours,hierarchy = cv2.findContours(norm_edges_closing, cv2.RETR_EXTERNAL                           #finding the contours in the frame (outer contours only)
                                                  , cv2.CHAIN_APPROX_SIMPLE)

    left,contours=left_contour(contours)
    
    ratio= compute_ratio(left,20.0)

    i=0
    if len(contours)!=0:                                                                                        #checking if there's any contours in the image
        for cnt in contours:
            coordinates.append([])
            rect = cv2.minAreaRect(cnt)                                                                         #calculating the distance of the contour form the center of the image  
            box = cv2.boxPoints(rect)
            box = np.int0(box)                                                                        
            angle=rect[2] *-1                                                                                   #calculating the angle at which the contour is rotated
            angle=round(angle,3)
            angles.append(angle)                                                                                #saving the angles and the coordinates difference into 2 lists
            coordinates[i].append(round((x/2-rect[0][0]),3))
            coordinates[i].append(round((y/2-rect[0][1]),3))
            coordinates_in_mm=np.round_(np.multiply(coordinates,ratio),3)
            coordinates_string=str(coordinates[i])
            angle=str(angle)
            cv2.putText(recoloured_frame,angle,                                                                 #drawing the angle text under the ic
                        (box[0][0],box[0][1]+25),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            cv2.putText(recoloured_frame,coordinates_string,
                        (box[0][0]-50,box[0][1]+50),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            cv2.drawContours(recoloured_frame,[box],0,(0,0,255),1)                                              #drawing the contours on the returned frame
            i+=1


    cv2.imshow('captured frame',frame)                                                                          #showing the processes for illustration
    cv2.imshow('recoloured',recoloured_frame)
    cv2.imshow('norm edges',norm_edges)
    cv2.imshow('frame_closing',frame_closing)
    cv2.imshow('no shadow',norm_img)
    cv2.imshow('norm_edges_closing',norm_edges_closing)
    cv2.imshow('gray',gray_frame)
    return recoloured_frame,coordinates_in_mm,angles    
    


#test code
cap=cv2.VideoCapture(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
out=cv2.VideoWriter('Video sample.avi',fourcc,20.0,(width,height))
while(True):
    ret,frame= cap.read()
    img,difference,angles=detect_ic_position_angle(frame)
    #cv2.imshow('image',img)  
    out.write(img)
    if cv2.waitKey(100)& 0xFF== ord('q'):
        
        break     
cap.release()
out.release()
cv2.destroyAllWindows()
