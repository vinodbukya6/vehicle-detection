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
# garbage collector and multiprocessing
import gc
from multiprocessing import Process, Manager
# yolo
from ultralytics import YOLO
from util_functions.night_vision import apply_night_vision, night_vision_core

model_path = "best_integer_quant_2cls.tflite"
#loading a YOLO model
model = YOLO(model_path, task='detect')

def feed_write(stack, cam, top: int) -> None:
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    # Original informations of video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(height, width, fps)
    while True:
        ret, img = cap.read()
        if ret:
            stack.append(img)
            #print("1")
        if len(stack) >= top:
            del stack[:]
            gc.collect()

def feed_read(stack) -> None:
    print('Process to read: %s' % os.getpid())
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
    cy_linha = 300 #500 #int(1500 * scale_percent/100 ) # modify
    cx_linha = 700 # roi => point1(0,cy_linha) ----------- point2(cx_linha, cy_linha)
    cx_sentido = int(2000 * scale_percent/100) # modify
    offset = 50 #int(30 * scale_percent/100 ) # modify
    contador_in = 0
    contador_out = 0

    # empty dataframe with empty values
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]
    df = pd.DataFrame({'Date_and_Time': [dt_string], 'Detections': [contador_out_plt], 'Total_objects_crossed': contador_out})
    roi_frame = True
    while True:
        if len(stack) != 0:
            frame = stack.pop()
            # night vision
            frame = night_vision_core(frame)
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            # Getting predictions
            results = model.predict(frame, imgsz=320, conf = 0.7,
                                    classes = class_IDS, device = 'cpu', verbose = False)
            # to understan ROI line - write one frame only
            if roi_frame:
                # Drawing transition line for in\out vehicles counting
                cv2.line(frame, (0, cy_linha), (cx_linha, cy_linha), (255,255,0), 2)
                cv2.line(frame, (0, cy_linha+offset), (cx_linha, cy_linha+offset), (255,255,0), 1)
                cv2.imwrite("roi_frame.jpg", frame)
                roi_frame = False

            # Get the boxes, and classes
            boxes = results[0].boxes.xywh.cpu()
            classes = results[0].boxes.cls.cpu().numpy()
            for box, class_id in zip(boxes, classes):
                x, y, w, h = box # xmin, ymin, xmax, ymax
                center_x, center_y = int(((x+w))/2), int((y+ h)/2)
                if (center_y < (cy_linha + offset)) and (center_y > (cy_linha - offset)):# Assuming objects cross horizontally
                    contador_out += 1
                    class_id = int(class_id)
                    veiculos_contador_out[class_id] += 1

            #print(dt_string, boxes, classes)

            # dictionary values
            contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]
            # new values
            new_row = {'Date_and_Time': dt_string, 'Detections': contador_out_plt, 'Total_objects_crossed': contador_out}
            #print(new_row)
            new_df = pd.DataFrame([new_row])
            # Check if the new row already exists in the DataFrame df empty initialized above
            # compare detections only for new entry we are writing in csv
            existing_row1 = df[df['Date_and_Time'] == dt_string] # total_objects_crossed
            existing_row2 = df[df['Total_objects_crossed'] == contador_out]
            
            # Add the new row if it does not exist
            if existing_row2.empty and existing_row1.empty:
                #df = df.append(new_row, ignore_index=True)
                df = pd.concat([df, new_df], ignore_index=True)
                #print("Predictions: ", dt_string, contador_out_plt)
                file_name_temp="detections_" + now.strftime("%d_%m_%Y") + ".csv"
                #file_name_temp = 'predicted_classes2.csv'
                # remove existing file
                if os.path.exists(file_name_temp):
                    os.remove(file_name_temp)
                # write same new file
                df.to_csv(file_name_temp, mode = "a", index=False, header=False)

                
    #Releasing the video
    cap.release()
    cv2.destroyAllWindows()
    temp = "Inference Completed!"
    return temp

if __name__ == '__main__':
    q = Manager().list()
    url = "rtsp://admin:rr123456@172.16.254.200:554/Streaming/Channels/1"  # Replace with your camera URL
    pw = Process(target=feed_write, args=(q, url, 100))
    pr = Process(target=feed_read, args=(q,))
    pw.start()
    pr.start()
    pr.join()
    pw.terminate()
    pr.terminate()



