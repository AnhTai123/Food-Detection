import os
import tempfile
import logging
from flask import Flask, request, render_template_string, send_file
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import imagehash
import torch
import numpy as np
from pyngrok import ngrok

# ---------------------------
# Cấu hình logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# ---------------------------
# Khởi tạo Flask app
# ---------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('instance', 'results')  # Thư mục lưu ảnh kết quả
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------
# Lookup table calories
# ---------------------------
calories_table = {
    'cake': 350, 'chicken curry': 300, 'croissant': 250, 'french fries': 312,
    'fried chicken': 400, 'fried rice': 220, 'hamburger': 600, 'noodles': 190,
    'pasta': 200, 'pizza': 800, 'roast chicken': 250, 'waffle': 290
}

# ---------------------------
# Tải mô hình YOLO
# ---------------------------
use_half = torch.cuda.is_available()
logging.info(f"Using half precision: {use_half}")
model = YOLO('./best.pt')

# ---------------------------
# HTML template nâng cấp
# ---------------------------
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Food Calorie Estimator</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background: #f4f4f9; }
        input[type="file"] { margin: 10px; }
        img { max-width: 80%; height: auto; margin-top: 20px; border-radius: 10px; }
        .result { margin-top: 20px; font-size: 16px; }
        table { border-collapse: collapse; margin: 0 auto; background: white; }
        th, td { border: 1px solid #ccc; padding: 8px 12px; }
        th { background-color: #f0f0f0; }
        #loading { display: none; color: blue; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Upload Image to Estimate Calories</h1>
    <form method="post" enctype="multipart/form-data" 
          onsubmit="document.getElementById('loading').style.display='block'">
        <input type="file" name="file" accept="image/*" required>
        <input type="submit" value="Upload">
    </form>
    <div id="loading">Đang xử lý...</div>
    {% if image_url %}
        <h2>Results:</h2>
        <img src="{{ image_url }}" alt="Processed Image">
        <div class="result">{{ results | safe }}</div>
    {% elif results %}
        <h2>Results:</h2>
        <div class="result">{{ results | safe }}</div>
    {% endif %}
</body>
</html>
"""

# ---------------------------
# Hàm lọc box trùng nâng cao
# ---------------------------
def merge_boxes(boxes, confs, classes, iou_threshold=0.5, center_threshold=0.2):
    """
    Hợp nhất các box trùng class khác nhau:
    - Loại box có confidence thấp hơn nếu IoU cao hoặc tâm gần nhau
    """
    keep = []
    used = set()

    # Duyệt theo confidence giảm dần
    for i in np.argsort(-confs):
        if i in used:
            continue
        keep.append(i)
        used.add(i)

        xi1, yi1, xi2, yi2 = boxes[i]
        cx1, cy1 = (xi1 + xi2)/2, (yi1 + yi2)/2
        w1, h1 = xi2 - xi1, yi2 - yi1

        for j in range(len(boxes)):
            if j in used:
                continue
            xj1, yj1, xj2, yj2 = boxes[j]
            cx2, cy2 = (xj1 + xj2)/2, (yj1 + yj2)/2
            w2, h2 = xj2 - xj1, yj2 - yj1

            # IoU
            xx1, yy1 = max(xi1,xj1), max(yi1,yj1)
            xx2, yy2 = min(xi2,xj2), min(yi2,yj2)
            inter = max(0, xx2-xx1) * max(0, yy2-yy1)
            union = w1*h1 + w2*h2 - inter
            iou_val = inter/union if union>0 else 0

            # Khoảng cách tâm
            dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            max_dim = max(max(w1,h1), max(w2,h2))
            center_close = dist < center_threshold*max_dim

            # Nếu overlap hoặc center gần nhau => loại box j
            if iou_val>iou_threshold or center_close:
                used.add(j)

    return keep

# ---------------------------
# Hàm xử lý upload & inference
# ---------------------------
def process_image(file) -> tuple[str, str]:
    try:
        img = Image.open(file.stream)
        img.verify()
        file.stream.seek(0)
    except Exception:
        return None, "Invalid image file"

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img_path = tmp.name
        file.save(img_path)

    img_hash = str(imagehash.average_hash(Image.open(img_path)))
    hash_csv_path = os.path.join('instance', 'cache', f'{img_hash}.csv')
    hash_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{img_hash}.jpg')
    os.makedirs(os.path.dirname(hash_csv_path), exist_ok=True)

    # Cache
    if os.path.exists(hash_csv_path) and os.path.exists(hash_img_path):
        df = pd.read_csv(hash_csv_path)
        total_calories = df['Calories'].sum()
        results_html = df[['Food', 'Calories', 'Confidence']].to_html(index=False)
        results_html += f"<br><b>Total Calories: {total_calories}</b>"
        os.remove(img_path)
        return f"/images/{img_hash}.jpg", results_html

    # YOLO inference
    results = model.predict(source=img_path, conf=0.5, iou=0.45, imgsz=320, half=use_half, save=True)
    if not results[0].boxes:
        os.remove(img_path)
        return None, "No food detected"

    # Lấy dữ liệu
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    names = results[0].names

    # Lọc box trùng nâng cao
    keep_idx = merge_boxes(boxes, confs, classes, iou_threshold=0.5, center_threshold=0.2)

    # DataFrame kết quả
    csv_data = []
    for idx in keep_idx:
        x1, y1, x2, y2 = boxes[idx]
        conf = confs[idx]
        cls = classes[idx]
        food_name = names[cls]
        calories = calories_table.get(food_name, 0)
        csv_data.append([food_name, calories, conf, x1, y1, x2, y2])

    df = pd.DataFrame(csv_data, columns=['Food', 'Calories', 'Confidence', 'x1', 'y1', 'x2', 'y2'])
    df.to_csv(hash_csv_path, index=False)

    # Lưu ảnh kết quả
    latest_img = max([os.path.join(r.save_dir, f) for r in results for f in os.listdir(r.save_dir) if f.endswith('.jpg')],
                     key=os.path.getctime)
    os.rename(latest_img, hash_img_path)

    # HTML kết quả
    total_calories = df['Calories'].sum()
    results_html = df[['Food', 'Calories', 'Confidence']].to_html(index=False)
    results_html += f"<br><b>Total Calories: {total_calories}</b>"

    os.remove(img_path)
    return f"/images/{img_hash}.jpg", results_html

# ---------------------------
# Routes Flask
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template_string(html_template, image_url=None, results="No file uploaded")
        image_url, results_text = process_image(file)
        return render_template_string(html_template, image_url=image_url, results=results_text)

    return render_template_string(html_template, image_url=None, results="")

@app.route('/images/<filename>')
def serve_image(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# ---------------------------
# Chạy ngrok và Flask
# ---------------------------
if __name__ == '__main__':
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        public_url = ngrok.connect(5000)
        print(f"Public URL: {public_url}")

    app.run(host='0.0.0.0', port=5000)
