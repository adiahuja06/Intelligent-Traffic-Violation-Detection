from sort.sort import Sort
import numpy as np
import cv2
import util
mot_tracker=Sort()
Sort()
from ultralytics import YOLO
#step 1 includes detecting the cars as we have to do object detection first and then we will be looking into the tracking part of it
coco_model=YOLO('yolov8n.pt')
#after we have detected the car we will be detecting the license plate so we have a pretrained model by using which i will be detecting the license plate
license_plate_detector=YOLO('best.pt')


#load the video
cap=cv2.VideoCapture('videos_car/cars.mp4')
vehicles=[2,3,5,7] #coco dataset has many class id but we need only car,motorbike,truck and bus so we wrote its indexes in the arra
results = {}
results_tesseract={}
frame_nmr=-1
ret=True
while ret and frame_nmr<120:
    frame_nmr+=1
    ret,frame=cap.read()
    if ret:
        results[frame_nmr] = {}
        #detect vehicles
        detections=coco_model(frame)[0]
        detections_=[] #in this we will be storing the bounding box of all the vehicles that are detected
        #print(detections)
        for detection in detections.boxes.data.tolist():
            #print(detection) so this will print the bounding box with the parameter x1,y1,x2,y2,score and class_id the x1 and y1 represent the top left corner bounding box and x2 and y2 represent the bottom right corner bounding box
            x1,y1,x2,y2,score,class_id=detection
            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2,score]) #we will store the bounding box which are in either of these categories namely car,truck,bus and train
        #track vehicles 
        track_ids=mot_tracker.update(np.asarray(detections_)) #so we need to have a unique id for the vehicles with bounding box detail so this function will perform that task 
        #now we will be detecting the license plates
        license_plates=license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=license_plate
            #now we need to assign the license plate to the car
            xcar1,ycar1,xcar2,ycar2,car_id=util.get_car(license_plate,track_ids)
            if car_id!=-1:
                print(car_id)
                #crop the license plate
                license_plate_crop=frame[int(y1):int(y2),int(x1):int(x2), :] #cropping the image where 
                #process license plat
                license_plate_crop_gray=cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh=cv2.threshold(license_plate_crop_gray,64,255,cv2.THRESH_BINARY_INV)
#                 #the below will show us all the images that we are detecting
                cv2.imshow('original_crop',license_plate_crop)
                cv2.imshow('threshold',license_plate_crop_thresh)
                print(frame_nmr)
                
                fin_img1=util.preprocess_image(license_plate_crop)
                #read the license plate number using easyOCR
                license_plate_text,license_plate_text_score=util.read_license_plate(fin_img1)
                #read the license plate using tesseractOCR
                fin_img2_thresh=util.thresh_preprocess(license_plate_crop)
                util.read_plate_tessearct(fin_img2_thresh)
                """
                #read license plate using tesseract ocr
                license_plate_text,license_plate_text_score=util.read_license_tesseract(license_plate_crop_thresh,license_plate_crop_gray)
                """
                cv2.imshow('preprocessed_image',fin_img1)
                cv2.imshow('preprocessed_thresh',fin_img2_thresh)
                cv2.waitKey(500)

                
                
                
                 #write the results
#                 if license_plate_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},   #this will store the data of frame and car id and we can read the license plate
                                                       'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                

# write results
util.write_csv(results,'./test4.csv')




"""
https://www.youtube.com/watch?v=y1ZrOs9s2QA&list=PLMoSUbG1Q_r8jFS04rot-3NzidnV54Z2q&index=15
Try the above code for building ocr from raw

"""
