# -*- coding: utf-8 -*-
import cv2
import video
import winsound as beep
import numpy as np
import win32api, win32con
from win32api import GetSystemMetrics


MIN_CALIB_MOMENT = 5
MIN_CURSOR_MOMENT = 10
SMOOTHING_ALPHA = 0.35
LOST_FRAME_GRACE = 5

if __name__ == '__main__':
    
    cv2.namedWindow("original")
    cv2.namedWindow("ir")
    #cv2.namedWindow("result")
    
    cap = video.create_capture(0)
    
    height = GetSystemMetrics(1) 
    width = GetSystemMetrics(0)
    
    cap.set(3, width)
    cap.set(4, height)
    
    ax = [0,0,0,0]
    ay = [0,0,0,0]
    
    i = 0
    j = False
    click = False
    smoothed_x = None
    smoothed_y = None
    lost_frames = LOST_FRAME_GRACE
    x = 0
    y = 0

    while True:
    
        flag, img = cap.read()
        
        low = np.array((0,0,255), np.uint8)
        
        high = np.array((180,0,255), np.uint8)
        
        try:
            
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img_hsv, low, high)
            mask = cv2.medianBlur(mask, 5)
            moments = cv2.moments(mask, 1)
            dM01 = moments['m01']
            dM10 = moments['m10']
            dArea = moments['m00']
            
            if i<4 and dArea >= MIN_CALIB_MOMENT:
                beep.Beep(1000,50) 
                if i>0:
                    if abs(ax[i-1]-int(dM10 / (dArea+0.0001)))> 100 or abs(ay[i-1]-int(dM01 / (dArea+0.0001)))> 100:
                       
                        ax[i] = int(dM10 / dArea)
                        ay[i] = int(dM01 / dArea)
                        i=i+1
                else:
                    ax[i] = int(dM10 / dArea)
                    ay[i] = int(dM01 / dArea)
                    i=i+1

            cv2.circle(img, (ax[0], ay[0]), 3,(0,255,255) , -1)
            cv2.circle(img, (ax[1], ay[1]), 3, (0,255,255), -1)
            cv2.circle(img, (ax[2], ay[2]), 3, (0,255,255), -1)
            cv2.circle(img, (ax[3], ay[3]), 3, (0,255,255), -1)
                        

            
            cv2.imshow('original', img)
            cv2.imshow('ir', mask)
            if i > 3:
                
                if j == False:
                    pts1 = np.float32 ( [ [ ax[0], ay[0] ], [ ax[1], ay[1] ], [ ax[2], ay[2] ], [ ax[3], ay[3] ] ] ) 
                    pts2 = np.float32 ( [ [ 0, 0 ], [ width, 0], [ 0, height ], [ width, height ] ] ) 
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    j = True
                            
                result = cv2.warpPerspective( mask, matrix, ( width, height ) )
                #cv2.imshow('ir', result)
                moments = cv2.moments(result, 1)
                dM01 = moments['m01']
                dM10 = moments['m10']
                dArea = moments['m00']
                    
                if dArea != 0:
                    x = int(dM10 / dArea)
                    y = int(dM01 / dArea)
                    win32api.SetCursorPos( ( x, y ) )
                    win32api.mouse_event( win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0 )
                    click = True                                                                       
                else:
                
                    if (x != 0 or y != 0) and click:
                        click = False
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
        except:
            cap.release()
            raise
        
        ch = cv2.waitKey(5)
        if ch == 27:
            break
    cap.release
    cv2.destroyAllWindows()
    
