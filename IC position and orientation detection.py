import cv2
import numpy as np
cap=cv2.VideoCapture(1)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('Video with lines.avi',fourcc,20.0,(640,480))

def detect_ic_position_angle(img):
    gray_frame=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    y,x=gray_frame.shape
    print x,y
    recoloured_frame=cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)
   

    cv2.line(recoloured_frame,((x)/2,0),((x)/2,y-1),(0,255,255),2)
    cv2.line(recoloured_frame,(0,y/2),(x-1,y/2),(0,255,255),2)
    #cv2.line(recoloured_frame,((x),0),((x),y-1),(0,255,255),2)

    
 
    kernel = np.ones((29,29),np.uint8)
    frame_closing=cv2.morphologyEx(gray_frame,cv2.MORPH_CLOSE,kernel)
    median_blured=cv2.medianBlur(frame_closing,11)
    
    diff_img=255- cv2.absdiff(gray_frame,median_blured)
    cv2.imshow('diff',diff_img)

    norm_img=gray_frame.copy()
    
    norm_img=cv2.normalize(diff_img,norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    norm_edges=cv2.Canny(diff_img,100,200)
    norm_edges_closing=cv2.morphologyEx(norm_edges,cv2.MORPH_CLOSE,kernel)
    norm_edges_closing=cv2.medianBlur(norm_edges_closing,15)

    image,contours,hierarchy = cv2.findContours(norm_edges_closing, cv2.RETR_EXTERNAL
                                                  , cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)!=0:
        for cnt in contours:
    
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            angle=rect[2]+90
            angle=str(angle)
            cv2.putText(recoloured_frame,angle,(box[0][0],box[0][1]+25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            cv2.drawContours(recoloured_frame,[box],0,(0,0,255),1)
    cv2.imshow('captured frame',frame)
    
    cv2.imshow('recoloured',recoloured_frame)
   
    cv2.imshow('norm edges',norm_edges)
    cv2.imshow('frame_closing',frame_closing)
    cv2.imshow('no shadow',norm_img)
    cv2.imshow('norm_edges_closing',norm_edges_closing)
    cv2.imshow('gray',gray_frame)
        
    


while(True):
    #ret,frame= cap.read()
    frame=cv2.imread('ic.png')
    contours=detect_ic_position_angle(frame)
       
    
    if cv2.waitKey(100)& 0xFF== ord('q'):
        
        break
      
cap.release()
out.release()
cv2.destroyAllWindows()
