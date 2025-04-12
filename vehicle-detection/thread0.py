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
# threading
import queue
import threading
q=queue.Queue()

# yolo
from ultralytics import YOLO
from util_functions.night_vision import apply_night_vision, night_vision_core

#loading a YOLO model
model_path = "best_integer_quant_2cls.tflite"
#loading a YOLO model
model = YOLO(model_path, task='detect')

def Receive():
    print("start Reveive")
    cap1 = cv2.VideoCapture("rtsp://admin:rr123456@192.168.1.88:554/Streaming/Channels/1")
    # Original informations of video
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    print(height, width, fps)
    # frame reading
    ret, frame = cap1.read()
    q.put(frame)
    while ret:
        #ret, frame = cap1.read()
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string, "read")
        q.put(frame)


def Display():
    print("Start Displaying")
    # variabels initializations
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
    cy_linha = int(1500 * scale_percent/100 )
    cx_sentido = int(2000 * scale_percent/100)
    offset = int(8 * scale_percent/100)
    contador_in = 0
    contador_out = 0

    # empty dataframe with empty values
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]
    df = pd.DataFrame({'Date_and_Time': [dt_string], 'Detections': [contador_out_plt], 'total_objects_crossed': contador_out})

    while True:
        if q.empty() !=True:
            print("1")
            frame=q.get()
            # night vision
            frame = night_vision_core(frame)
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            # Getting predictions
            results = model.predict(frame, imgsz=320, conf = 0.7,
                                    classes = class_IDS, device = 'cpu', verbose = False) #classes - class_IDS

            # Get the boxes, track IDs, and classes
            boxes = results[0].boxes.xywh.cpu()
            classes = results[0].boxes.cls.cpu().numpy() #y_hat[0].boxes.cls.cpu().numpy()
            print(dt_string, classes)
            # iterate through detections
            for box, class_id in zip(boxes, classes):
                x, y, w, h = box # xmin, ymin, xmax, ymax
                center_x, center_y = int(((x+w))/2), int((y+ h)/2)
                # ROI line crossed or not
                if (center_y < (cy_linha + offset)) and (center_y > (cy_linha - offset)):
                    contador_out += 1
                    class_id = int(class_id)
                    veiculos_contador_out[class_id] += 1

        # dictionary values
        contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]
        #print(dt_string,classes)
        # new values
        new_row = {'Date_and_Time': dt_string, 'Detections': contador_out_plt, 'total_objects_crossed': contador_out}
        new_df = pd.DataFrame([new_row])
        # Check if the new row already exists in the DataFrame df empty initialized above
        # compare detections only for new entry we are writing in csv
        existing_row = df[df['total_objects_crossed'] == contador_out]

        # Add the new row if it does not exist
        if existing_row.empty:
            df = pd.concat([df, new_df], ignore_index=True)
            print("Predictions: ", dt_string, contador_out_plt)
            file_name_temp = 'predicted_classes_6th.csv'
            # remove existing file
            if os.path.exists(file_name_temp):
                os.remove(file_name_temp)

            # write same new file
            df.to_csv(file_name_temp, mode = "a", index=False, header=False)

if __name__=='__main__':
    
    p1=threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()

#if __name__ == '__main__':
    
    # Example usage:
 #   rtsp_video_feed = '/home/vinod/Projects/VehicleDetection/videosClient/video4.mkv'
  #  vehicledetctionsystem(rtsp_video_feed)

