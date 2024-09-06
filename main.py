import numpy as np
import cv2
import sys
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

# Load a model
model = YOLO('yolov8s.pt')  # pretrained YOLOv8n model
names = model.names

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
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = 120,
                                                        decisionThreshold = 0.8)
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
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=1*30, #  Khắc phục tình trạng quá nhạy, tốc độ cập nhật background
                                                        useHistory = False, # Nên để False, để True thì cập nhật rất chậm
                                                        maxPixelStability = 50*30, # Chỉ có tác dụng khi useHistory = True 
                                                        isParallel=True)
    '''
    A lower value for maxPixelStability means the algorithm will reinitialize pixels as part of the background more frequently, allowing quicker adaptation to changes in the scene.
    A higher value for maxPixelStability means pixels will be considered part of the background for a more extended period before reinitialization, making the algorithm more resistant to changes.
    '''
    print('Invalid detector!')
    sys.exit(0)


# Cắt frame, đầu vào là các điểm neo ( keypoints ) từ vùng ROI
def crop_frame_based_on_roi(frame, roi_points):
    # Extract ROI coordinates
    x, y, width, height = roi_points
    
    # Crop the frame based on the ROI
    cropped_frame = frame[y:y + height, x:x + width]
    return cropped_frame

bg_subtractor = getBGSubtractor(BGS_TYPE)
minLocalArea = 250 # Độ thay đổi nhỏ nhất của các vùng nhỏ (contour) để được xem xét
 

def main():

    cap = cv2.VideoCapture('rtsp://admin:Admin123@192.168.10.64:554/Streaming/Channels/101/')

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
    roi_points = cv2.selectROI("Select ROI", frame_background, fromCenter=False, showCrosshair=True)
    x_roi, y_roi, width_roi, height_roi = roi_points
    total_area = width_roi * height_roi
    minGlobalArea = total_area * 0.1 # Độ thay đổi nhỏ nhất của toàn cục để quyết định có chuyển động

    # Close the window after ROI selection
    cv2.destroyWindow("Select ROI")

    # Calculate the four points representing the vertices of the ROI
    point1 = [x_roi, y_roi]
    point2 = [x_roi + width_roi, y_roi]
    point3 = [x_roi + width_roi, y_roi + height_roi]
    point4 = [x_roi, y_roi + height_roi]

    roi_draw =  np.array([point1, point2, point3, point4],np.int32)
    roi_draw = roi_draw.reshape((-1, 1, 2))


    while (cap.isOpened):
        globalArea = 0
        ok, frame = cap.read()
        if frame is not None:
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
                results = model.predict(frame_in_roi,conf = 0.7,verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        c = box.cls
                        class_name  = model.names[int(c)]
                        prob = float(box.conf)
                        print(f'Detected {class_name} with {prob}')
                        x1_c,y1_c,x2_c,y2_c = box.xyxy[0]
                        x1_c,y1_c,x2_c,y2_c = int(x1_c+x_roi), int(y1_c+y_roi), int(x2_c+x_roi), int(y2_c+y_roi)
                        
                        
                        cv2.rectangle(frame,(x1_c,y1_c),(x2_c,y2_c),(0,0,255),thickness=2)
                        cv2.putText(frame,class_name + ' ' + str(prob) ,(x1_c,y1_c),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),2)

            roi_color = (0, 255, 0)  # Green color for the ROI

            frame = cv2.polylines(frame, [roi_draw], isClosed=True, color=roi_color, thickness=2)

            
            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print('Frame is None')
            print(cap.isOpened())

main()