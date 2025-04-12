import os
import shutil
import cv2
import random
# from base64 import b64encode
from PIL import Image as PILImage
import torch
import time
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import csv

# yolo
from ultralytics import YOLO
from util_functions.night_vision import apply_night_vision, night_vision_core

model_path = "best_integer_quant_2cls.tflite"
#loading a YOLO model
model = YOLO(model_path, task='detect')


def vehicledetctionsystem(rtsp_video_feed):
    # input request file
    input_file = rtsp_video_feed
    
    if input_file:
        
        cap1 = cv2.VideoCapture(input_file)

        # Objects to detect Yolo
        #class_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
        #dict_classes = {0: 'auto', 1: 'bicycle', 2: 'bus', 3: 'car', 4: 'tempo', 5: 'tractor', 6: 'two_wheelers', 7: 'vehicle_truck'}
        #class_labels = ['auto', 'bicycle', 'bus', 'car', 'tempo', 'tractor', 'two_wheelers', 'vehicle_truck']
        class_IDS = [0, 1]
        class_labels = ["motor_vehicles", "two_wheelers"] #2 finalised
        dict_classes = {0: 'motor_vehicles', 1: 'two_wheelers'}

        scale_percent = 50
        # Auxiliary variables
        centers_old = {}
        centers_new = {}
        obj_id = 0
        veiculos_contador_in = dict.fromkeys(class_IDS, 0)
        veiculos_contador_out = dict.fromkeys(class_IDS, 0)
        end = []
        frames_list = []
        cy_linha = 750 #int(1500 * scale_percent/100 ) # modify
        cx_sentido = int(2000 * scale_percent/100) # modify
        offset = int(8 * scale_percent/100 ) # modify
        contador_in = 0
        contador_out = 0

        # Store the track history
        track_history = defaultdict(lambda: [])
        # Create a dictionary to keep track of objects that have crossed the line
        crossed_objects_dict = {}

        # Original informations of video
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap1.get(cv2.CAP_PROP_FPS)
        print(height, width, fps)
        ### Video output ####
        video_name = 'result.avi'
        output_path = "rep_" + video_name
        tmp_output_path = "./temp_outputs/" + "track2.avi"

        # empty dataframe with empty values
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]
        df = pd.DataFrame({'Date_and_Time': [dt_string], 'Detections': [contador_out_plt], 'total_objects_crossed': contador_out})

        
        temp_frame = 0
        average_fps = 0.0
        
        while True:
            # read 4 video streams separately
            ret, frame = cap1.read()
            if not ret:
                print("Breakkkkkk")
                break

            # skip frames. Per second process only 4 frames
            #if count%7.5 == 0:
            if True:
                #print(count, "count")
                # original stream -> 1080 * 1920, resize it to nearest resolution
                #frame = cv2.resize(frame, (640, 640))

                # single inference
                #task = main_inference(frame1,exec_net,yolo_layer_params,net,labels_map,count)

                # night vision
                frame = night_vision_core(frame)
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                # Getting predictions
                #y_hat = model.predict(frame, imgsz=320, conf = 0.7, classes = class_IDS, device = 'cpu', verbose = False) #classes - class_IDS
                results = model.track(frame, imgsz=320, conf = 0.7, classes = class_IDS, device = 'cpu', verbose = False,
                                    persist=True, save=False, tracker="bytetrack.yaml")

                # Get the boxes, track IDs, and classes
                boxes = results[0].boxes.xywh.cpu()
                #track_ids = results[0].boxes.id.int().cpu().tolist()
                track_ids = results[0].boxes.id.cpu().tolist() if results[0].boxes.id is not None else []
                #print("track_ids", track_ids)
                classes = results[0].boxes.cls.cpu().numpy() #y_hat[0].boxes.cls.cpu().numpy()
    
                annotated_frame = frame

                # Plot the tracks and count objects crossing the line

                for box, track_id, class_id in zip(boxes, track_ids, classes):
                    x, y, w, h = box # xmin, ymin, xmax, ymax
                    center_x, center_y = int(((x+w))/2), int((y+ h)/2)
                    
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    #cy_linha = int(1500 * scale_percent/100 )
                    #cx_sentido = int(2000 * scale_percent/100)
                    

                    # Check if the object crosses the line
                    if 0 < x < int(4500 * scale_percent/100) and abs(y - cy_linha) < 5:
                    #if (center_y < (cy_linha + offset)) and (center_y > (cy_linha - offset)):# Assuming objects cross horizontally
                        if track_id not in crossed_objects_dict:
                            contador_out += 1
                            class_id = int(class_id)
                            veiculos_contador_out[class_id] += 1

            count+=1 # for processing less frames
            # dictionary values
            contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]
            # new values
            new_row = {'Date_and_Time': dt_string, 'Detections': contador_out_plt, 'total_objects_crossed': contador_out}
            new_df = pd.DataFrame([new_row])
            #df = pd.DataFrame({'Date_and_Time': [dt_string], 'Detections': [contador_out_plt], 'total_objects_crossed': contador_out})
            # Check if the new row already exists in the DataFrame df empty initialized above
            # compare detections only for new entry we are writing in csv
            existing_row = df[df['total_objects_crossed'] == contador_out]

            # Add the new row if it does not exist
            if existing_row.empty:
                #df = df.append(new_row, ignore_index=True)
                df = pd.concat([df, new_df], ignore_index=True)
                file_name_temp = 'predicted_classes2.csv'
                # remove existing file
                if os.path.exists(file_name_temp):
                    os.remove(file_name_temp)
                # write same new file
                df.to_csv(file_name_temp, mode = "a", index=False, header=False)


                
        #Releasing the video
        cap1.release()
        cv2.destroyAllWindows()
        temp = "Inference Completed!"
        return temp

if __name__ == '__main__':
    
    # Example usage:
    rtsp_video_feed = "rtsp://admin:123456@192.168.1.100:554"
    vehicledetctionsystem(rtsp_video_feed)


