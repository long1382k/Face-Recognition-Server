import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Sử dụng hàm
image_path = "static/images_directory/002/NLong.png"  # Thay đường dẫn bằng đường dẫn tới ảnh của bạn
base64_string = image_to_base64(image_path)
print(base64_string)
print(len(base64_string))
