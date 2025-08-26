import cv2 as cv
import mediapipe as mp
import time
import math
import os
import pandas as pd

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.p_time = 0

    def find_hands(self, frame, draw=True):
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_positions(self, frame):
        positions = [] 
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                hand_pos = []
                h, w, _ = frame.shape
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_pos.append((id, cx, cy))
                positions.append(hand_pos)
        return positions

    def get_fps(self):
        c_time = time.time()
        fps = 1 / (c_time - self.p_time) if (c_time - self.p_time) != 0 else 0
        self.p_time = c_time
        return int(fps)
    
    def get_hand(self, frame):
      h, w, _ = frame.shape    
      if self.results.multi_hand_landmarks and self.results.multi_handedness:
        for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
            label = handedness.classification[0].label  # "Right" or "Left"
            
            # Get position of wrist (landmark 0)
            cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
            
            # Draw label on frame
            cv.putText(frame, label, (cx - 40, cy - 40), cv.FONT_HERSHEY_SIMPLEX, 1, (244, 45, 23), 2)
            
            return label  # Or return (label, cx, cy) if needed
    


       
    def fingers_up(self,frame,draw=True):
        positions=HandTracker.get_positions(self,frame)
        fingers=[]
        for hands in positions:
            fingers_up=0
            # Check each finger
            if hands:
              side=HandTracker.get_hand(self,frame)
              # Thumb: check x instead of y because it bends inward
              if side=="Right":
                if hands[4][1]< hands[3][1]:
                   fingers_up+=1
                   
              else:
                if hands[4][1]>hands[3][1]:
                   fingers_up+=1
              
              # Other 4 fingers: compare tip and joint y-coordinates
              for tip_id in [8, 12, 16, 20]:
                if hands[tip_id][2] < hands[tip_id - 2][2]:
                    fingers_up +=1

                        
            if draw==True:
                cv.putText(frame, f'{side} hand: {fingers_up}', (50,400),cv.FONT_HERSHEY_SIMPLEX,1, (0,0,155), 2)        
            return fingers_up
        
    def distance(self,frame,draw=True):
     positions=HandTracker.get_positions(self,frame)
     for hands in positions:
        for id,cx,cy in hands:
            if len(hands)!=0:
                # print(hands[4],hands[8])
                
                x1,y1=hands[8][1],hands[8][2]
                x2,y2=hands[12][1],hands[12][2]
                cx,cy=(x1+x2)//2,(y1+y2)//2
                length=math.hypot(x2-x1,y2-y1)
                if draw==True:
                    cv.circle(frame,(x1,y1),10,(218,220,47),cv.FILLED)
                    cv.circle(frame,(x2,y2),10,(218,220,47),cv.FILLED)
                    cv.circle(frame,(cx,cy),10,(218,220,47),cv.FILLED)
                    cv.line(frame,(x1,y1),(x2,y2),(218,220,47),2)
                    if length<40:
                       cv.circle(frame,(cx,cy),10,(154,46,62),cv.FILLED)    
                return  length       
            
            
            
class HandDataCollector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)

    def store_hands(self, frame, all_data):
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            landmark_row = []
            for lm in hand_landmarks.landmark:
                landmark_row.extend([lm.x, lm.y, lm.z])
            all_data.append(landmark_row)
        return all_data

    def extract_from_folder(self, folder_path, output_csv="output.csv"):
        all_data = []
        labels = []

        for label in os.listdir(folder_path):
            class_path = os.path.join(folder_path, label)
            if not os.path.isdir(class_path):
                continue

            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                frame = cv.imread(img_path)
                if frame is None:
                    print(f"Failed to read {img_path}")
                    continue
                all_data, labels = self.store_hands(frame, all_data, labels, img_path, label)

        # Save to CSV
        columns = [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]
        df = pd.DataFrame(all_data, columns=columns)
        df['label'] = labels
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} samples to {output_csv}")
           