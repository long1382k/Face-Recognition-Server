# Install Flask on your system by writing
#!pip install Flask
#Import all the required libraries
#Importing Flask
#render_template--> To render any html file, template

from flask import Flask, Response,jsonify,request,stream_with_context

# Required to run the YOLOv8 model
import cv2
from ultralytics import YOLO
import json
import time
import os
import threading
import urllib3
from urllib.parse import urlencode
import base64

# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
# from YOLO_Video import video_detection
app = Flask(__name__)




predicting = dict()
streaming = dict()


import numpy as np
import cv2
import sys
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

# Load a model
#names = model.names

TEXT_COLOR = (0, 255, 0)
TRACKER_COLOR = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = "videos/Cars.mp4"

BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
# 2 ok
BGS_TYPE = BGS_TYPES[4]

# Hậu xử lý
def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation": # làm dày cạnh
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == "opening": # erosion + dilation = reduce noise outside
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == "closing": # dilation + erosion   = reduce noise inside
        kernel = np.ones((3, 3), np.uint8)

    return kernel
# Chọn các hàm tương ứng với các hậu xử lý
def getFilter(img, filter):
    if filter == 'closing': 
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)

    if filter == 'opening': 
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)

    if filter == 'dilation':
        return cv2.dilate(img, getKernel("dilation"), iterations=2)

    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, getKernel("dilation"), iterations=2)

        return dilation

# Chọn thuật toán trừ nền 
def getBGSubtractor(BGS_TYPE):
    # https://docs.opencv.org/3.4/d1/d5c/classcv_1_1bgsegm_1_1BackgroundSubtractorGMG.html

    # https://docs.opencv.org/3.4/d6/da7/classcv_1_1bgsegm_1_1BackgroundSubtractorMOG.html
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history = 10, nmixtures = 5,
                                                       backgroundRatio = 0.8, noiseSigma=0)
    # https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history = 500, detectShadows = True,
                                                  varThreshold = 100)
    # https://docs.opencv.org/3.4/db/d88/classcv_1_1BackgroundSubtractorKNN.html
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history = 500, dist2Threshold=400,
                                                 detectShadows = True)
    # https://docs.opencv.org/3.4/de/dca/classcv_1_1bgsegm_1_1BackgroundSubtractorCNT.html
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=2*30, #  Khắc phục tình trạng quá nhạy, tốc độ cập nhật background
                                                        useHistory = False, # Nên để False, để True thì cập nhật rất chậm
                                                        maxPixelStability = 50*30, # Chỉ có tác dụng khi useHistory = True 
                                                        isParallel=True)
    '''
    A lower value for maxPixelStability means the algorithm will reinitialize pixels as part of the background more frequently, allowing quicker adaptation to changes in the scene.
    A higher value for maxPixelStability means pixels will be considered part of the background for a more extended period before reinitialization, making the algorithm more resistant to changes.
    '''
    print('Invalid detector!')
    sys.exit(0)

http = urllib3.PoolManager()
def send_image(image, obj_id, camid, detect_time, type,api_code):
    retval, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer)
    raw_params = {'image': encoded_image}
    params = urlencode(raw_params)
    # api code: Loai xu ly 1=Face 2=Fire 3=Car 4=Intrusion
    #type 1=crop obj  2=frame
    infor_param = urlencode(
        {'obj_id': obj_id, 'camid': 3, 'detect_time': detect_time, 'type': type, 'api_code': api_code})
    # page = 'http://192.168.10.65:8015/Jobs/Upload?' + infor_param
    page = 'http://192.168.10.65:8015/Jobs/Upload?' + infor_param
    request = http.request('POST', page, body=params, headers={
                        'Content-type': 'application/x-www-form-urlencoded; charset=UTF-8'})
    #print(request.data)

# Cắt frame, đầu vào là các điểm neo ( keypoints ) từ vùng ROI
def crop_frame_based_on_roi(frame, roi_points):
    # Extract ROI coordinates
    x, y, width, height = roi_points
    
    # Crop the frame based on the ROI
    cropped_frame = frame[y:y + height, x:x + width]
    return cropped_frame

bg_subtractor = getBGSubtractor(BGS_TYPE)
minLocalArea = 50 # Độ thay đổi nhỏ nhất của các vùng nhỏ (contour) để được xem xét


def get_detection(cam_id):
    global predicting
    while True:
        if predicting[cam_id][1] == False:
            break
        while str('Frame') not in predicting[cam_id][2]:
            time.sleep(0.5)
        frame = predicting[cam_id][2]['Frame']
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break


def gen_motion_detect(cam_id):
    model = YOLO('yolov8s.pt')  # pretrained YOLOv8n model
    global predicting
    cap = predicting[cam_id][0]

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return
    sucess, frame_background = cap.read()

    if not sucess:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # Draw ROI on the background frame (first frame)
    roi_points = (316, 191, 723, 403)
    x_roi, y_roi, width_roi, height_roi = roi_points
    total_area = width_roi * height_roi
    minGlobalArea = total_area * 0.1 # Độ thay đổi nhỏ nhất của toàn cục để quyết định có chuyển động


    # Calculate the four points representing the vertices of the ROI
    point1 = [x_roi, y_roi]
    point2 = [x_roi + width_roi, y_roi]
    point3 = [x_roi + width_roi, y_roi + height_roi]
    point4 = [x_roi, y_roi + height_roi]

    roi_draw =  np.array([point1, point2, point3, point4],np.int32)
    roi_draw = roi_draw.reshape((-1, 1, 2))

    count = 0
    while (True):
        globalArea = 0
        ret, frame = cap.read()
        start_time = time.time()
        if not ret: 
            print('Cannot read frame ')
            break
        

        if frame is not None:
        # Remove the oldest frame to maintain the buffer size

            frame_in_roi = crop_frame_based_on_roi(frame, roi_points) # Crop frame in ROI
            bg_mask = bg_subtractor.apply(frame_in_roi) # Áp dụng bộ lọc
            bg_mask = getFilter(bg_mask, 'opening') # Hậu xử lý
            bg_mask = cv2.medianBlur(bg_mask, 5) # Làm mờ 

            (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Tìm các vùng liên thông ( contours) với tuỳ chọn RETR_EXTERNAL 
            x1 = x_roi + width_roi
            x2 = x_roi
            y1 = y_roi + height_roi
            y2 = y_roi
            motion_detected = False
            for cnt in contours: # Xét từng vùng nhỏ 
                localArea = cv2.contourArea(cnt) # Tính diện tích của phần thay đổi
                
                if localArea >= minLocalArea: # Nếu diện tích lớn hơn giá trị ngưỡng thì:
                    globalArea = globalArea + localArea # Cộng dồn vào phần thay đổi tổng thể
                    x_rect, y_rect, w, h = cv2.boundingRect(cnt) # Vị trí tương đối so với phần ROI
                    x_rect = x_rect + x_roi
                    y_rect = y_rect + y_roi
                
                    x1 = min(x1, x_rect)
                    x2 = max(x_rect + w, x2)
                    y1 = min(y1, y_rect)
                    y2 = max(y_rect + h, y2)   

            if globalArea > minGlobalArea:
                    #cv2.rectangle(frame, (10,30), (250,55), (255,0,0), -1)
                cv2.putText(frame, 'Motion detected!', (10,50), FONT, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)
                motion_detected = True

            result = cv2.bitwise_and(frame_in_roi, frame_in_roi, mask=bg_mask)
            if motion_detected:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 1)
                results = model.predict(frame_in_roi,conf = 0.7,verbose=False,classes=[0])
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        #c = box.cls
                        class_name  = r.names[box.cls[0].item()]
                        prob = float(box.conf)
                        print(f'Detected {class_name} with {prob}')
                        
                        # send image 
                        detected_time = time.time()
                        cropped_image = frame_in_roi
                        thread_send_image1 = threading.Thread(target=send_image,args =(cropped_image,class_name,'cam_id',detected_time,1,4,))
                        thread_send_image1.start()

                        x1_c,y1_c,x2_c,y2_c = box.xyxy[0]
                        x1_c,y1_c,x2_c,y2_c = int(x1_c+x_roi), int(y1_c+y_roi), int(x2_c+x_roi), int(y2_c+y_roi)
                        
                        cv2.rectangle(frame,(x1_c,y1_c),(x2_c,y2_c),(0,0,255),thickness=2)
                        cv2.putText(frame,class_name + ' ' + str(prob) ,(x1_c,y1_c),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),2)

            roi_color = (0, 255, 0)  # Green color for the ROI
            frame = cv2.polylines(frame, [roi_draw], isClosed=True, color=roi_color, thickness=2)

            predict_results = dict()
            predict_results['Frame'] = frame
            predicting[cam_id][2] = predict_results   

    cap.release()
    cv2.destroyAllWindows() 


@app.route('/motion_detect/<path:cam_id>',methods=["GET"])
def motion_detect(cam_id):
    global predicting
    if str(cam_id) in predicting:
        print('Existed')
        if predicting[cam_id][1] == False:
            predicting[cam_id][1] = True
    else:
        print('Creating new...')
        cap = cv2.VideoCapture(cam_id)
        predicting[cam_id] = [cap,True,dict()]
        thread_gen_detecion = threading.Thread(target=gen_motion_detect,args =(cam_id,))
        thread_gen_detecion.start()
    return 'success'

@app.route('/show/<path:cam_id>')
def show(cam_id):
    cam_lab = 'rtsp://admin:Admin123@192.168.10.64/Src/MediaInput/h264/stream_1/ch_' 
    cam_id = cam_lab
    generator = get_detection(cam_id)
    return Response(generator, mimetype='multipart/x-mixed-replace; boundary=frame')
# STOP PREDICT

@app.route('/stoppredict/<path:cam_id>',methods=["GET"])
def stop(cam_id):
    global predicting
    #predicting[cam_id][0].release()
    predicting[cam_id][1] = False
    time.sleep(0.5)
    del predicting[cam_id]
    return('Stopped predict on ' + cam_id)

    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0',port=5003)
