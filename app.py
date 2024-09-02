from flask import Flask, Response, render_template
import cv2
from imageio.v3 import imwrite
from insightface.app import FaceAnalysis
from matplotlib.style.core import available
from pyparsing import White
from qdrant_client import QdrantClient, models
import uuid
from flask import Flask, request, jsonify
import numpy as np
import os
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
client = QdrantClient(url="http://localhost:6333")
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './static/images_directory/'  # Đường dẫn cố định để lưu ảnh
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
face_app = FaceAnalysis(providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))


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

def generate_frames():
    collection_name = 'registed_face'
    count = 0
    camera = cv2.VideoCapture(0)  # Capture from webcam
    while True:
        success, frame = camera.read()
        frame = resize_frame(frame, width=480)
        if not success:
            break
        else:
            count+=1
            # Perform object detection here using YOLO
            if count % 3 == 0:
                faces = face_app.get(frame)
                for face in faces:
                    face_embedding = face.embedding

                    # Query the Qdrant database
                    search_result = client.search(
                        collection_name=collection_name,
                        query_vector=face_embedding,
                        score_threshold=0.6,
                        limit=1  # Adjust the limit based on your need,
                    )

                    if not search_result:
                        print({'message': 'No similar face found'})
                    else:

                    # Return the closest match from the database
                        print(search_result[0].payload['name'])


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
        student_fullName = request.form['fullName']
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

@app.route('/video_feed')
def video_feed():
    generate_frames()

@app.route('/count_student', methods=['GET', 'POST'])
def count_student():
    collection_name = 'registed_face'
    available_students = []
    camera_rtsp = request.form['camera_rtsp']
    if not camera_rtsp:
        return jsonify({'error': 'camera_rtsp is required'}), 400
    camera = cv2.VideoCapture(camera_rtsp)  # Capture from webcam

    while True:
        # Read 1 frame
        success, frame = camera.read()
        if not success:
            return jsonify({"success": False, "message": "Failed to read frame"}), 400

        # Recognition
        faces = face_app.get(frame)
        cv2.imwrite('abc.jpg', frame)
        if faces:
            print(f'Found {len(faces)} faces')
            break
    for face in faces:
        face_embedding = face.embedding
        # Query the Qdrant database
        search_result = client.search(
            collection_name=collection_name,
            query_vector=face_embedding,
            score_threshold=0.6,
            limit=1  # Adjust the limit based on your need,
        )

        if search_result:
            available_students.append(search_result[0].payload)
    return available_students


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088, debug=False)
