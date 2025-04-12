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
# yolo
from ultralytics import YOLO
from util_functions.night_vision import apply_night_vision, night_vision_core

#loading a YOLO model
model = YOLO('last.pt') #yolov8n.pt, best.pt

#geting names from classes
dict_classes = model.model.names
#print(dict_classes)

def vehicledetctionsystem(rtsp_video_feed):
    # input request file
    input_file = rtsp_video_feed
    
    if input_file:
        
        cap1 = cv2.VideoCapture(input_file)
        #cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Objects to detect Yolo
        class_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
        #{0: 'auto', 1: 'bicycle', 2: 'bus', 3: 'car', 4: 'tempo', 5: 'tractor', 6: 'two_wheelers', 7: 'vehicle_truck'}

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
        offset = int(8 * scale_percent/100 )
        contador_in = 0
        contador_out = 0

        # Original informations of video
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap1.get(cv2.CAP_PROP_FPS)
        print(height, width, fps)
        ### Video output ####
        video_name = 'result.avi'
        output_path = "rep_" + video_name
        tmp_output_path = "./temp_outputs/" + filename.split(".")[0]+ ".avi"

        fourcc=cv2.VideoWriter_fourcc(*'XVID')
        output_video=cv2.VideoWriter(tmp_output_path,fourcc,fps,(width,height))


        #fps_start_time = time.time() # for fps, start time =0
        count = 1
        
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

                temp_start = time.time()
                # Getting predictions
                y_hat = model.predict(frame, conf = 0.7, classes = class_IDS, device = 'cpu', verbose = False)
                temp_end = time.time()
                temp_frame = temp_frame + 1
                fps = 1.0/(temp_end-temp_start)
                print("FPS: ", "%.1f" % fps, "Infernce time in sec: ", 1/fps)
                average_fps += fps

                # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
                boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
                conf    = y_hat[0].boxes.conf.cpu().numpy()
                classes = y_hat[0].boxes.cls.cpu().numpy()

                # Storing the above information in a dataframe
                #positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
                # Extract data from the 'boxes' object
                data = {'xmin': boxes[:, 0], 'ymin': boxes[:, 1],
                        'xmax': boxes[:, 2], 'ymax': boxes[:, 3],
                        'conf': conf, 'class': classes}

                # Create DataFrame
                positions_frame = pd.DataFrame(data)

                #Translating the numeric class labels to text
                labels = [dict_classes[i] for i in classes]

                # Drawing transition line for in\out vehicles counting
                cv2.line(frame, (0, cy_linha), (int(4500 * scale_percent/100 ), cy_linha), (255,255,0), 8)

                # For each vehicles, draw the bounding-box and counting each one the pass thought the transition line (in\out)
                for ix, row in enumerate(positions_frame.iterrows()):
                    # Getting the coordinates of each vehicle (row)
                    xmin, ymin, xmax, ymax, confidence, category,  = row[1].astype('int')

                    # Calculating the center of the bounding-box
                    center_x, center_y = int(((xmax+xmin))/2), int((ymax+ ymin)/2)

                    # drawing center and bounding-box of vehicle in the given frame
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 5) # box
                    cv2.circle(frame, (center_x,center_y), 5,(255,0,0),-1) # center of box

                    #Drawing above the bounding-box the name of class recognized.
                    cv2.putText(img=frame, text=labels[ix]+' - '+str(np.round(conf[ix],2)),
                                org= (xmin,ymin-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)

                    # Checking if the center of recognized vehicle is in the area given by the transition line + offset and transition line - offset
                    if (center_y < (cy_linha + offset)) and (center_y > (cy_linha - offset)):
                        if  (center_x >= 0) and (center_x <=cx_sentido):
                            contador_in +=1
                            veiculos_contador_in[category] += 1
                        else:
                            contador_out += 1
                            veiculos_contador_out[category] += 1

                #updating the counting type of vehicle
                contador_in_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_in.items()]
                contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]

                #drawing the number of vehicles in\out
                cv2.putText(img=frame, text='N. vehicles In',
                            org= (30,30), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                            fontScale=1, color=(255, 255, 0),thickness=1)

                cv2.putText(img=frame, text='N. vehicles Out',
                            org= (int(2800 * scale_percent/100 ),30),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=1)

                #drawing the counting of type of vehicles in the corners of frame
                xt = 40
                for txt in range(len(contador_in_plt)):
                    xt +=30
                    cv2.putText(img=frame, text=contador_in_plt[txt],
                                org= (30,xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=1, color=(255, 255, 0),thickness=1)
                    cv2.putText(img=frame, text=contador_out_plt[txt],
                                org= (int(2800 * scale_percent/100 ),xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=1, color=(255, 255, 0),thickness=1)

                #drawing the number of vehicles in\out
                cv2.putText(img=frame, text=f'In:{contador_in}',
                            org= (int(1820 * scale_percent/100 ),cy_linha+60),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=2)

                cv2.putText(img=frame, text=f'Out:{contador_out}',
                            org= (int(1800 * scale_percent/100 ),cy_linha-40),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=2)

                #Saving frames in a list
                frames_list.append(frame)
                #saving transformed frames in a output video formaat
                output_video.write(frame)
                
            # increase the frame count
            count += 1

        print("Average FPS: %.1f" % (average_fps / temp_frame), "Total Frames: ", temp_frame)       
        #fps_end_time = time.time()
        #time_diff = fps_end_time - fps_start_time
        #print(time_diff, "time diff")
        
        #Releasing the video
        output_video.release()
        cap1.release()
        cv2.destroyAllWindows()
        temp = "Inference Completed!"
        return temp

if __name__ == '__main__':
    
    # Example usage:
    rtsp_video_feed = ''
    vehicledetctionsystem(rtsp_video_feed)

