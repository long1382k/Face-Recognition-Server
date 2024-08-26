from flask import Flask
import cv2
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient, models
import uuid
from flask import Flask, request, jsonify
import numpy as np
import os
client = QdrantClient(url="http://localhost:6333")
app = Flask(__name__)
UPLOAD_FOLDER = './static/images_directory/'  # Đường dẫn cố định để lưu ảnh
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def creae_collection(collection_name):
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
        except Exception as e:
            print(e)
    else:
        print(f'Collection {collection_name} already exists')
@app.route('/add_face', methods=['GET', 'POST'])
def add_face():
    # Initialize the InsightFace model
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # list of student to save to database
    points = []

    # Define the name of the collection
    collection_name = 'registed_face'
    # check if collection is existed else create new
    creae_collection(collection_name)

    # Lấy các thông tin từ form
    student_id = request.form['id']
    student_name = request.form['name']
    student_class = request.form['class']
    # Lấy file ảnh từ request
    student_image = request.files['image']

    # Kiểm tra và lưu file vào thư mục cố định
    if student_image:
        filename = student_image.filename
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], student_id), exist_ok=True)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{student_id}/{filename}')
        student_image.save(image_path)
        student = dict()
        student = {
            "id": student_id, # mã học viện
            "name": student_name, # tên học viên
            "class": student_class, # lớp
            "image": image_path # đường dẫn ảnh
        }
        img = cv2.imread(image_path)

        # Detect faces
        faces = face_app.get(img)
        print(len(faces))
        if len(faces) > 0:
            face_embedding = faces[0].embedding
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector= face_embedding.tolist(),  # Vector đã mã hóa
                payload= student  # Dữ liệu gốc của đối tượng `student`
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
    return jsonify({"success": True, "message": f"Added student {student_name} to database"}), 200


@app.route('/check_image', methods=['GET', 'POST'])
def check_image():
    return 'Hello world'
@app.route('/search', methods=['GET', 'POST'])
def search_face():
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
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
    faces = app.get(img)
    if not faces:
        return jsonify({'error': 'No face detected'}), 400
    face_embedding = faces[0].embedding

    # Query the Qdrant database
    search_result = client.search(
        collection_name=collection_name,
        query_vector=face_embedding,
        limit=1  # Adjust the limit based on your need
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
@app.route('/test_sample', methods=['POST','GET'])
def test_sample():
    # Initialize the InsightFace model
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Define the name of the collection
    collection_name = 'registed_face'
    # Create a collection if it doesn't exist
    creae_collection(collection_name)
    students = [
        {
            "id": "001", # mã học viện
            "name": "Nguyen Huu Long", # tên học viên
            "class": "ATTT", # lớp
            "image": "./static/images_directory/001/HLong.png" #
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
            id= str(uuid.uuid4()),
            vector=face_embedding.tolist(),  # Vector đã mã hóa
            payload= student  # Dữ liệu gốc của đối tượng `student`
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=False)
