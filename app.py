from imageio.v3 import imwrite
from insightface.app import FaceAnalysis
from matplotlib.style.core import available
from pyparsing import White
from qdrant_client import QdrantClient, models
import numpy as np
import os
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import queue
import threading

from unidecode import unidecode

#new import
from flask import Flask, render_template, Response, request, jsonify, stream_with_context
from aiortc import RTCPeerConnection, RTCSessionDescription
import cv2
import json
import uuid
import asyncio
import logging
import time

# Required to run the YOLOv8 model
from ultralytics import YOLO
import json
import urllib3
from urllib.parse import urlencode
import base64

client = QdrantClient(url="http://localhost:6333")
app = Flask(__name__,static_url_path='/static')
CORS(app)

UPLOAD_FOLDER = './static/images_directory/'  # Đường dẫn cố định để lưu ảnh
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model_name = 'buffalo_sc'  # Example model name
face_app = FaceAnalysis(name=model_name,providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
predicting = dict()
streaming = dict()
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

# motion detect
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


#  end motion detect functions
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
                        # thread_send_image1 = threading.Thread(target=send_image,args =(cropped_image,class_name,'cam_id',detected_time,1,4,))
                        # thread_send_image1.start()

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



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Lấy tham số camera_rtsp từ query string
    camera_rtsp = request.args.get('camera_rtsp')
    print(camera_rtsp, "2")
    if not camera_rtsp:
        return "camera_rtsp parameter is required", 400

    # Trả về phản hồi được tạo bởi hàm generate_frames với camera_rtsp
    return Response(generate_frames(camera_rtsp),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def create_new_collection(collection_name):
    if not client.collection_exists(collection_name):
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=512,  # Vector size is defined by used model
                    distance=models.Distance.COSINE,
                ),
            )
            print(f'Created collection {collection_name}')
            # return jsonify({"success": True, "message": f"Created collection {collection_name}"}), 200

        except Exception as e:
            return False
            # return jsonify({f'error': 'Failed to create collection {collection_name}'}), 400
            # print(f'Failed to create collection {collection_name}')

    else:
        print(f'Collection {collection_name} already exists')
    return True
def resize_frame(frame, width=800):
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))
def generate_frames(camera_rtsp):
    student_set = set()
    collection_name = 'registed_face'
    count = 0
    #camera_rtsp = 'rtsp://blueai:Blueai1234@nsvanphuc.ddns.net:554/streaming/channels/1401'
    #camera_rtsp = 'https://drive.google.com/uc?export=download&id=16SrD_X3RkdG0rmt0hREOUiwNN5HsiSFH'
    #camera_rtsp = 'https://drive.google.com/uc?export=download&id=18Ayy8rqwsB99kZiIh8ysk4M55OGuppZr'
    # video trang
    #camera_rtsp='https://drive.google.com/uc?export=download&id=1WXSEu0Bke9sDAyzu7hmqADQbzN7pNteC'
    print(camera_rtsp)
    camera = cv2.VideoCapture('static/video/video2.mp4')  # Capture from webcam
    while True:
        process_start = time.time()
        success, frame = camera.read()
        #frame = resize_frame(frame, width=480)
        if not success:
            break
        else:
            count+=1
            # Perform object detection here using YOLO
            if count % 2 ==0:
                faces = face_app.get(frame)
                for face in faces:
                    face_embedding = face.embedding
                    #
                    # # Query the Qdrant database
                    search_result = client.search(
                        collection_name=collection_name,
                        query_vector=face_embedding,
                        score_threshold=0.4,
                        limit=1  # Adjust the limit based on your need,
                    )
                    #
                    # if not search_result:
                    #     print({'message': 'No similar face found'})
                    # else:
                    #
                    # # Return the closest match from the database
                    #     print(search_result[0].payload['name'])
                    box = face.bbox.astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    # Put the processed frame into the queue
                    if search_result:
                        cv2.putText(frame,search_result[0].payload['fullName'],(box[0],box[1]),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                        student_set.add(search_result[0].payload['fullName'])
                    print("Available students:",student_set)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # concat frame one by one and show result
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            process_end = time.time()
            process_time = process_end - process_start
            #print(f"Frame generation time: {process_time} seconds")

        #     box = face.bbox.astype(int)
            #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            #         # for landmark in face.landmark_2d_106:
            #         #     cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)
            #         # gender, age = face.gender, int(face.age)
            #         # label = f"{'Male' if gender == 1 else 'Female'}, {age}"
            #         # cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            #
            # ret, buffer = cv2.imencode('.jpg', frame)
            # frame = buffer.tobytes()
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
async def offer_async():
    params = await request.json
    offer_var = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create an RTCPeerConnection instance
    pc = RTCPeerConnection()

    # Generate a unique ID for the RTCPeerConnection
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc_id = pc_id[:8]

    # Create and set the local description
    await pc.createOffer()
    await pc.setLocalDescription(offer_var)

    # Prepare the response data with local SDP and type
    response_data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return jsonify(response_data)
def offer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    future = asyncio.run_coroutine_threadsafe(offer_async(), loop)
    return future.result()
# Route to handle the offer request
@app.route('/offer', methods=['POST'])
def offer_route():
    return offer()

@app.route('/add_student', methods=['GET', 'POST'])
def add_student():

    # list of student to save to database
    points = []
    # Define the name of the collection
    collection_name = 'registed_face'
    # check if collection is existed else create new
    if create_new_collection(collection_name):
        # Lấy các thông tin từ form
        student_id = request.form['id']
        student_fullName = unidecode(request.form['fullName']) # chuyen chuoi Tieng Viet sang form Tieng Anh
        student_className = request.form['className']
        student_classId = request.form['classId']

        # Lấy file ảnh từ request
        student_image = request.files['image']

        # Kiểm tra và lưu file vào thư mục cố định
        if student_image:
            class_directory = os.path.join(app.config['UPLOAD_FOLDER'], f'{student_classId}')

            if not os.path.exists(class_directory):
                os.makedirs(class_directory)
            # Get the directory where images for this student are stored
            student_directory = os.path.join(app.config['UPLOAD_FOLDER'], f'{student_classId}/{student_id}')
            # Ensure the directory exists
            os.makedirs(student_directory, exist_ok=True)

            # Get the list of existing files to determine the next available integer
            existing_files = os.listdir(student_directory)

            # Filter out any non-numeric filenames and get the highest number
            numbers = [int(os.path.splitext(f)[0]) for f in existing_files if f.split('.')[0].isdigit()]
            next_number = max(numbers, default=0) + 1  # Start from 1 if no files are present

            # Define the new filename using the next available number
            new_filename = f"{next_number}{os.path.splitext(student_image.filename)[1]}"  # Preserve the file extension

            # Save the image with the new filename
            image_path = os.path.join(student_directory, new_filename)
            student_image.save(image_path)

            student = dict()
            student = {
                "id": student_id,  # mã học viện
                "fullName": student_fullName,  # tên học viên
                "className": student_className,  # lớp
                "classId": student_classId,
                "image": image_path  # đường dẫn ảnh
            }
            img = cv2.imread(image_path)

            # Detect faces
            faces = face_app.get(img)
            if len(faces) > 0:
                face_embedding = faces[0].embedding
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=face_embedding.tolist(),  # Vector đã mã hóa
                    payload=student  # Dữ liệu gốc của đối tượng `student`
                )
                points.append(point)
                client.upload_points(
                    collection_name=collection_name,
                    points=points,
                )
            else:
                return jsonify({'error': 'No face detected'}), 400
        else:
            return jsonify({'error': 'No image provided'}), 400
        # Count the number of records in the collection
        count_result = client.count(collection_name)
        record_count = count_result.count
        print({f'Uploaded {len(points)} students to {collection_name} database. Total of records: {record_count}'})
        return jsonify({"success": True, "message": f"Added student {student_fullName} to database"}), 200
    else:
        return jsonify({'error': 'Failed to create collection'}), 400

@app.route('/delete_student', methods=['GET', 'POST'])
def delete_student():
    student_id = request.form['id']

@app.route('/get_all_students', methods=['GET', 'POST'])
def get_all_students():
    # Specify the collection name
    collection_name = 'registed_face'

    results = client.scroll(
        collection_name=f"{collection_name}",
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    records = results[0]

    data = []
    for record in records:

        student = record.payload
        data.append(student)
    # Dictionary to store grouped entries
    grouped_data = {}

    # Grouping entries by ID
    for entry in data:
        entry_id = entry['id']
        if entry_id in grouped_data:
            infor = grouped_data[entry_id][0]
            infor['image'] = [infor['image']]
            infor['image'].append(entry['image'])
            infor['image_count'] = len(infor['image'])
            grouped_data[entry_id] = infor
        else:
            entry['image_count'] = 1
            grouped_data[entry_id] = [entry]

    return jsonify(grouped_data), 200

@app.route('/check_image', methods=['GET', 'POST'])
def check_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image'].read()
    img = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Detect and extract face embedding
    faces = face_app.get(img)
    if not faces:
        return jsonify({'error': 'No face detected'}), 400
    return jsonify({"success": True, "message": f"Verified image"}), 200

@app.route('/search', methods=['GET', 'POST'])
def search_face():
    # Get the collection_name parameter from the form data
    collection_name = request.form.get('collection_name')
    if not collection_name:
        return jsonify({'error': 'collection_name is required'}), 400
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image'].read()
    img = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Detect and extract face embedding
    faces = face_app.get(img)
    if not faces:
        return jsonify({'error': 'No face detected'}), 400
    face_embedding = faces[0].embedding

    # Query the Qdrant database
    search_result = client.search(
        collection_name=collection_name,
        query_vector=face_embedding,
        score_threshold=0.6,
        limit=1  # Adjust the limit based on your need,
    )

    if not search_result:
        return jsonify({'message': 'No similar face found'}), 404


    # Return the closest match from the database
    return jsonify({
        'id': search_result[0].id,
        'score': search_result[0].score,
        'payload': search_result[0].payload
    })

@app.route('/delete_collection', methods=['DELETE'])
def delete_collection():
    # Get the collection name from the request
    collection_name = request.form.get('collection_name')
    if not collection_name:
        return jsonify({"error": "Collection name is required"}), 400
    if not client.collection_exists(collection_name):
        return jsonify({"success": False, "message": f"Collection '{collection_name}' not existed."})
    try:
        # Delete the collection
        client.delete_collection(collection_name)
        return jsonify({"success": True, "message": f"Collection '{collection_name}' deleted successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/test_sample', methods=['POST', 'GET'])
def test_sample():
    # Define the name of the collection
    collection_name = 'registed_face'
    # Create a collection if it doesn't exist
    if create_new_collection(collection_name):
        students = [
            {
                "id": "001",  # mã học viện
                "name": "Nguyen Huu Long",  # tên học viên
                "class": "ATTT",  # lớp
                "image": "./static/images_directory/001/HLong.png"  #
            },
            {
                "id": "002",
                "name": "Do Ngoc Long",
                "class": "KTPM",
                "image": "./static/images_directory/002/NLong.png"

            },
        ]
        points = []
        for idx, student in enumerate(students):
            img_path = student['image']
            img = cv2.imread(img_path)
            # Detect faces
            face_embedding = face_app.get(img)[0].embedding

            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=face_embedding.tolist(),  # Vector đã mã hóa
                payload=student  # Dữ liệu gốc của đối tượng `student`
            )
            points.append(point)
            client.upload_points(
                collection_name=collection_name,
                points=points,
            )
        # Count the number of records in the collection
        count_result = client.count(collection_name)
        record_count = count_result.count
        return f'Uploaded {len(points)} students to {collection_name} database. Total of records: {record_count}'
    else:
        return jsonify({'error': 'Failed to create collection'}), 400


@app.route('/count_student', methods=['GET', 'POST'])
def count_student():
    collection_name = 'registed_face'
    available_students = []
    camera_rtsp = request.form['camera_rtsp']
    if not camera_rtsp:
        return jsonify({'error': 'camera_rtsp is required'}), 400
    camera = cv2.VideoCapture(camera_rtsp)  # Capture from webcam

    while True:
        # Read 1 framef
        success, frame = camera.read()
        if not success:
            return jsonify({"success": False, "message": "Failed to read frame"}), 400

        # Recognition
        faces = face_app.get(frame)
        #cv2.imwrite('abc.jpg', frame)
        if faces:
            print(f'Found {len(faces)} faces')
            break
    for face in faces:
        face_embedding = face.embedding
        # Query the Qdrant database
        search_result = client.search(
            collection_name=collection_name,
            query_vector=face_embedding,
            score_threshold=0.2,
            limit=1  # Adjust the limit based on your need,
        )

        if search_result:
            available_students.append(search_result[0].payload)
    return available_students

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
    # cam_lab = 'rtsp://admin:Admin123@192.168.10.64/Src/MediaInput/h264/stream_1/ch_' 
    # cam_id = cam_lab
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088, debug=False)

