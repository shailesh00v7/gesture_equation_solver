import cv2 as cv
import time
import numpy as np
from hand_tracking_module import HandTracker
import autopy
from keras.models import load_model
import sympy as sp
import matplotlib.pyplot as plt
from join import digits

digit_rec=digits()
detector=HandTracker(detection_confidence=0.8)


alpha=0.25
prev_canvas = None
brushThickness=15
eraserThickness=50
frameR=100
imgcanvas=np.zeros((720,1200,3),np.uint8)
img_res=np.zeros((720,1200,3),np.uint8)
xp,yp=0,0
wscr,hscr=autopy.screen.size()
capture=cv.VideoCapture(0)
capture.set(3,wscr)
capture.set(4,hscr)
drawcolor=(255,255,255)


while True:
    success,frame=capture.read()
    frame=cv.flip(frame,1)
    
    cv.rectangle(frame,(150,0),(650,125),(255,255,255),cv.FILLED)
    cv.putText(frame,"pen",(350,60),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)
    cv.rectangle(frame,(750,0),(1250,125),(0,0,0),cv.FILLED)

    cv.putText(frame,"eraser",(950,60),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
    
    
    # find  hand landmarks
    frame=detector.find_hands(frame)
    positions=detector.get_positions(frame)
    for hands in positions:
        if len(hands)!=0:   
            # finding which fingers are up
            fingers=detector.fingers_up(frame)
            
            # selection mode-- if two fingers are up
            index_up = hands[8][2] < hands[6][2]
            middle_up=hands[12][2] < hands[10][2]
            middle_down = hands[12][2] > hands[10][2]
            ring_up = hands[16][2] < hands[14][2]
            ring_down = hands[16][2] > hands[14][2]
            pinky_up = hands[20][2] < hands[18][2]
            pinky_down = hands[20][2] > hands[18][2]
            
            
            # tip of index and middle finger
            x1,y1=hands[8][1:]
            x2,y2=hands[12][1:]
            
            if index_up and middle_up and ring_down and pinky_down:
                xp,yp=0,0
                if y1<125:
                    if 150<x1<650:
                        drawcolor=(255,255,255)
                    # elif 450<x1<650:
                    #     drawcolor=(0,0,255)
                    elif 750<x1<1250:
                        drawcolor=(0,0,0)
                    # elif 1050<x1<1250:
                    #     drawcolor=(0,0,0)        

                        
                cv.rectangle(frame,(x1,y1-25),(x2,y2+25),drawcolor,cv.FILLED)
                cv.putText(frame, f"Mode: select", (10, 40),cv.FONT_HERSHEY_SIMPLEX, 1, (100,0,155), 2)
                
            
            # drawing mode -- if one finger is up
            elif index_up and middle_down and ring_down and pinky_down:
                cv.circle(frame,(x1,y1),15,drawcolor,cv.FILLED)
                cv.putText(frame, f"Mode: drawing", (10, 40),cv.FONT_HERSHEY_SIMPLEX, 1, (100,0,155), 2)
                if xp==0 and yp==0:
                    xp,yp=x1,y1
                # Apply EMA smoothing
                x1 = int(alpha * x1 + (1 - alpha) * xp)
                y1 = int(alpha * y1 + (1 - alpha) * yp)
                if drawcolor==(0,0,0):
                    cv.line(frame,(xp,yp),(x1,y1),drawcolor,eraserThickness)
                    cv.line(imgcanvas,(xp,yp),(x1,y1),drawcolor,eraserThickness)  
                else:        
                    cv.line(frame,(xp,yp),(x1,y1),drawcolor,brushThickness)
                    cv.line(imgcanvas,(xp,yp),(x1,y1),drawcolor,brushThickness)    
                xp,yp=x1,y1
    
    
    imggray=cv.cvtColor(imgcanvas,cv.COLOR_BGR2GRAY)
    _,imginv=cv.threshold(imggray,50,255,cv.THRESH_BINARY_INV)
    # Resize imginv to match frame
    imginv = cv.resize(imginv, (frame.shape[1], frame.shape[0]))

    # If imgnv is single channel, convert to 3-channel BGR
    if len(imginv.shape) == 2 or imginv.shape[2] == 1:
        imginv = cv.cvtColor(imginv, cv.COLOR_GRAY2BGR)
    frame=cv.bitwise_and(frame,imginv)
    # Resize imgcanvas to match frame size
    imgcanvas = cv.resize(imgcanvas, (frame.shape[1], frame.shape[0]))

    # If imgcanvas is grayscale or has 1 channel, convert to BGR
    if len(imgcanvas.shape) == 2 or imgcanvas.shape[2] == 1:
        imgcanvas = cv.cvtColor(imgcanvas, cv.COLOR_GRAY2BGR)

    # Now apply bitwise OR safely
    frame = cv.bitwise_or(frame, imgcanvas)
    
    # fps
    fps=detector.get_fps()
    cv.putText(frame, f"FPS: {fps}", (10, 150), cv.FONT_HERSHEY_COMPLEX,1, (255, 0, 0), 2)
              
    cv.imshow("img",frame)
    cv.imshow("canvas",imgcanvas)
    key = cv.waitKey(1)
    if key==ord("s"):
        img_res=imgcanvas.copy()
        print("image saved.....")



        label_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                    10: '+', 11: '-', 12: '*', 13: '/', 14: '='}
        
                    # Extract character ROIs
        # Get full recognition
         

        equation, labels, char_images,boxes = digit_rec.recognize_expression(img_res)
        # # Reconstruct equation
        equation = ''.join(equation)
        print("Extracted Equation:", equation)
        try:
            result = sp.sympify(equation)  # Parses safely
            print(f"{equation} = {result}")
        except Exception as e:
            print("Error solving equation:", e)

        # ===== Show original image with bounding boxes =====
        img_vis = img_res.copy()
        for (x, y, w, h), label in zip(boxes, labels):
            cv.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(img_vis, str(label), (x, y - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        plt.figure(figsize=(8, 4))
        plt.imshow(cv.cvtColor(img_vis, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"extracted equation : {equation}\nResult : {result}", fontsize=16)
        plt.show()


       
        # --- Draw boxes on original image ---
        for (x, y, w, h), label in zip(boxes,labels):
            cv.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(img_res, label, (x, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

   
    if cv.waitKey(1) & 0xFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()    
