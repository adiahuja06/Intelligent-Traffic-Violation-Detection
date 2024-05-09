import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import time
import csv
from math import dist
model=YOLO('yolov8s.pt') #a pretrained model used for object detection if you want to make a custom model you will have to use pytorch

def RGB(event,x,y,flags,param): #this function helps to print the co ordinates of x and y as the mouse pointer moves
    if event==cv2.EVENT_MOUSEMOVE:
        colorsBGR=[x,y]
        print(colorsBGR) 

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB',RGB)

cap=cv2.VideoCapture('traffic2.mp4')
my_file=open("coco.txt","r")
data=my_file.read()
class_list=data.split("\n") #this will convert all the classes into the list so that once our class is detected we can easily print it on our rectangle
print(class_list)
count=0
tracker=Tracker() #importing the class from the other python file
ret=True
offset=6
cy1=324 #this is our y co-ordinate of the first line
cy2=370 #this is y co ordinate of the another line
vh={} #this is dictionary which will store the the id of the vehicle
counter=[]
frame_nmr=-1
cars_data={}
def add_car_entry(cars_dict, car_id, frame_number, speed):
    cars_dict[car_id] = {'frame_number': frame_number, 'speed': speed}
while ret:
    frame_nmr+=1
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1020,500))
    #results=model.predict(frame,show=True) if we put show=True then it means that it will show the bounding box around the cars that can be seen
    #now we need to learn how can we manually make the bounding box
    result=model.predict(frame)
    boxes = result[0].boxes.xyxy.tolist() #this will help is to get the bounding box information of all the cars detected
    classes = result[0].boxes.cls.tolist()
    names = result[0].names
    confidences = result[0].boxes.conf.tolist()
    px=pd.DataFrame(boxes).astype("float") 
    px1=pd.DataFrame(classes).astype("float") 
    px2=pd.DataFrame(confidences).astype("float") 
    final=pd.concat([px,px2,px1],axis=1,ignore_index=True) #this is the final dataframe that we have got ignore_index is done to ignore some of the indexes as it do not align with the dataframe
    list=[] #this is just to get the updated bounding box from the tracker file
    for index,row in final.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5]) #this will tell us the class detected as our data frame has the class number
        c=class_list[d] #this will fetch the name of the class at that index
        if 'car' in c: #so it will now only detect the cars and draw bounding box around it
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in list:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for bbox in bbox_id:
            x3,y3,x4,y4,id=bbox #here we are associating an id to each car that is unique
            cx=int(x3+x4)//2
            cy=int(y3+y4)//2
            if cy1<(cy+offset) and cy1>(cy-offset):
                vh[id]=time.time() #we will store the time at which the car touched the center point
            if id in vh:
                if cy2<(cy+offset) and cy2>(cy-offset):
                    elapsed_time=time.time()-vh[id]
                    if counter.count(id)==0: #this will check same car id should not be repeated twice
                        counter.append(id)
                        distance=cy2-cy1 #this will be distace between two lines
                        a_speed_ms=distance/elapsed_time
                        a_speed_kh=a_speed_ms*3.6
                        cv2.circle(frame,(cx,cy),4,(0,0,255),-1) #the co ordinates of cx and cy are the center point so if center point is in ROI we will detect the car 4 represents the radius and the next section represents the colour
                        cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
                        cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                        add_car_entry(cars_data,id,frame_nmr,a_speed_kh)
    cv2.line(frame,(311,cy1),(883,cy1),(255,255,255),1)
    cv2.putText(frame,str('Line1'),(315,319),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),2)
    cv2.line(frame,(284,370),(911,370),(255,255,255),1)
    cv2.putText(frame,str('Line2'),(291,365),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),2)
    dh=(len(counter))
    cv2.putText(frame,str('Count:'+str(dh)),(91,48),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)
    cv2.imshow("RGB",frame)
    if cv2.waitKey(1)&0xFF==ord('q'): #if we put cv2.waitKey(0) that will freeze the first frame and on the vs code you will be able to see all the co ordinates but if you do it 1 then you can run all the frames
        break
with open('cars_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['ID', 'Frame Number', 'Speed']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for car_id, car_info in cars_data.items():
        writer.writerow({'ID': car_id, 'Frame Number': car_info['frame_number'], 'Speed': car_info['speed']})
cap.release()
cv2.destroyAllWindows()

#the above code help us to draw the bounding boxes near our car but we need to count and track the car in the region of intrest that will help us to detect speed





'''
Flow of video for speed detection is as below
'''
