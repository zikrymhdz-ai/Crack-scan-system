from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime
import base64
from ultralytics import YOLO

app = Flask(__name__)

# ─────────────────────────────────────────────
# MODEL LOADING — YOLOv8 crack/damage detector
# ─────────────────────────────────────────────
# Priority order: best custom model → fallback to yolov8n
MODEL_CANDIDATES = [
    'wall_crack.pt',       # ✅ confirmed working
    'yolov8n.pt',          # fallback - auto downloads if wall_crack.pt missing
]

model = None
AI_READY = False
MODEL_NAME = "None"

for candidate in MODEL_CANDIDATES:
    try:
        model = YOLO(candidate)
        AI_READY = True
        MODEL_NAME = candidate
        print(f"✅ Model loaded: {candidate}")
        break
    except Exception as e:
        print(f"⚠️  Could not load {candidate}: {e}")

if not AI_READY:
    print("❌ No model loaded — edge detection only mode")

# ─────────────────────────────────────────────
# DATABASE — SQLite (works on Render, no setup)
# ─────────────────────────────────────────────
DB_PATH = 'crack_data.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS cracks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path  TEXT,
            crack_level INTEGER,
            crack_area  INTEGER,
            timestamp   TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────
# CRACK ANALYSIS CORE
# ─────────────────────────────────────────────
def analyze_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, 0, 0, None

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── EDGE DETECTION (Canny + Morphology) ──
    blur       = cv2.GaussianBlur(gray, (5, 5), 0)
    edges      = cv2.Canny(blur, 35, 130)
    kernel     = np.ones((3, 3), np.uint8)
    crack_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    edge_area  = int(np.sum(crack_mask > 0) / 4)

    # ── YOLO AI DETECTION ──
    ai_area    = 0
    ai_boxes   = []
    ai_confs   = []

    if AI_READY and model is not None:
        try:
            results = model(
                img_path,
                conf=0.20,          # lower threshold = more sensitive
                iou=0.45,
                imgsz=640,
                verbose=False
            )
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf  = float(box.conf[0])
                    w     = x2 - x1
                    h     = y2 - y1
                    ai_boxes.append((x1, y1, x2, y2))
                    ai_confs.append(conf)
                    # Area contribution proportional to box size + confidence
                    ai_area += int(w * h * conf * 0.4)

            # Also use segmentation mask if available (yolov8-seg models)
            if result.masks is not None:
                for mask in result.masks.data:
                    mask_np = mask.cpu().numpy()
                    ai_area += int(np.sum(mask_np > 0.5) * 0.5)

        except Exception as e:
            print(f"⚠️ YOLO inference error: {e}")

    # ── HYBRID SCORING ──
    # Weight AI more heavily when confident detections exist
    if ai_boxes:
        total_area = int(edge_area * 0.45 + ai_area * 1.55)
    else:
        total_area = int(edge_area * 0.85)

    # ── SEVERITY LEVEL ──
    if total_area < 50:
        level  = 0; status = "SAFE ✅"
    elif total_area < 2500:
        level  = 1; status = "Minor — Small cracks detected"
    elif total_area < 8000:
        level  = 2; status = "Medium — Inspection advised"
    else:
        level  = 3; status = "🚨 SEVERE — Immediate action required"

    # ── VISUALIZATION ──
    overlay = original.copy()

    # Red crack highlight from edge mask
    if level > 0:
        red_layer = original.copy()
        red_layer[crack_mask == 255] = [0, 0, 255]
        overlay = cv2.addWeighted(original, 0.55, red_layer, 0.55, 0)
    else:
        overlay = original.copy()

    # AI bounding boxes with confidence labels
    for (x1, y1, x2, y2), conf in zip(ai_boxes, ai_confs):
        box_color = (255, 0, 200)   # magenta for AI detections
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
        label = f"Crack {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(overlay, (x1, y1 - lh - 8), (x1 + lw + 6, y1), box_color, -1)
        cv2.putText(overlay, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Status banner at top
    banner_colors = {0: (0,180,60), 1: (0,200,200), 2: (0,140,255), 3: (0,0,220)}
    bcolor = banner_colors.get(level, (100,100,100))
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 52), bcolor, -1)

    level_labels = {0:"SAFE", 1:"MINOR", 2:"MEDIUM", 3:"SEVERE"}
    cv2.putText(overlay, f"{level_labels[level]}  |  Area: {total_area}px",
                (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    # AI model tag bottom-right
    tag = f"Model: {MODEL_NAME}" if AI_READY else "Edge Only"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    h_img, w_img = overlay.shape[:2]
    cv2.rectangle(overlay, (w_img - tw - 14, h_img - th - 16), (w_img, h_img), (0,0,0), -1)
    cv2.putText(overlay, tag, (w_img - tw - 8, h_img - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)

    return overlay, level, total_area, status


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        os.makedirs('uploads', exist_ok=True)
        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        # Accept base64 (live capture) or file upload
        if 'image_data' in request.form:
            img_data  = request.form['image_data']
            img_bytes = base64.b64decode(img_data.split(',')[1])
            img_path  = f'uploads/{filename}'
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
        else:
            file     = request.files['image']
            filename = file.filename or filename
            img_path = f'uploads/{filename}'
            file.save(img_path)

        result_img, level, total_area, status = analyze_image(img_path)

        if result_img is None:
            return "Error reading image", 400

        out_name = f"annotated_{os.path.splitext(os.path.basename(filename))[0]}.jpg"
        out_path = f'uploads/{out_name}'
        cv2.imwrite(out_path, result_img)

        # Save to DB
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO cracks (image_path, crack_level, crack_area, timestamp) VALUES (?,?,?,?)",
                (img_path, level, total_area, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB write error: {e}")

        return render_template('result.html',
                               level=level,
                               area=total_area,
                               status=status,
                               img=f'/uploads/{out_name}')

    # GET — show homepage with recent history
    history = []
    try:
        conn    = get_db()
        records = conn.execute(
            "SELECT image_path, crack_level, crack_area, timestamp FROM cracks ORDER BY timestamp DESC LIMIT 6"
        ).fetchall()
        conn.close()
        for r in records:
            history.append({
                'image_url': f'/uploads/{os.path.basename(r["image_path"])}',
                'level':     r['crack_level'],
                'area':      r['crack_area'],
                'timestamp': r['timestamp'][-8:-3] if r['timestamp'] else ''
            })
    except Exception as e:
        print(f"DB read error: {e}")

    return render_template('index.html', history=history)


@app.route('/live')
def live():
    return render_template('live.html', ai_ready=AI_READY, model_name=MODEL_NAME)


# ─────────────────────────────────────────────
# OPTION B — SERVER-SIDE AI VIDEO STREAM
# ─────────────────────────────────────────────
import threading
import time
from flask import Response, jsonify

# Shared stats updated by the frame generator
live_stats = {
    'level': 0, 'area': 0, 'status': 'SAFE',
    'ai_boxes': 0, 'coverage': '0.00', 'fps': 0,
}
stats_lock    = threading.Lock()
camera_instance = None
camera_lock   = threading.Lock()

def get_camera():
    global camera_instance
    with camera_lock:
        if camera_instance is None or not camera_instance.isOpened():
            camera_instance = cv2.VideoCapture(0)
            camera_instance.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            camera_instance.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera_instance.set(cv2.CAP_PROP_FPS, 30)
        return camera_instance

def release_camera():
    global camera_instance
    with camera_lock:
        if camera_instance is not None:
            camera_instance.release()
            camera_instance = None

def analyze_frame_live(frame):
    original = frame.copy()
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_img, w_img = frame.shape[:2]

    # Edge detection
    blur       = cv2.GaussianBlur(gray, (5,5), 0)
    edges      = cv2.Canny(blur, 35, 130)
    kernel     = np.ones((3,3), np.uint8)
    crack_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    edge_area  = int(np.sum(crack_mask > 0) / 4)
    coverage   = round((np.sum(crack_mask > 0) / (w_img * h_img)) * 100, 2)

    # YOLO AI
    ai_area  = 0
    ai_boxes = []
    ai_confs = []
    if AI_READY and model is not None:
        try:
            results = model(frame, conf=0.20, iou=0.45, imgsz=640, verbose=False)
            result  = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    ai_boxes.append((x1, y1, x2, y2))
                    ai_confs.append(conf)
                    ai_area += int((x2-x1)*(y2-y1)*conf*0.4)
        except Exception as e:
            print(f"Live YOLO error: {e}")

    # Hybrid score
    total_area = int(edge_area*0.45 + ai_area*1.55) if ai_boxes else int(edge_area*0.85)
    if   total_area < 50:   level = 0; status = "SAFE"
    elif total_area < 2500: level = 1; status = "MINOR"
    elif total_area < 8000: level = 2; status = "MEDIUM"
    else:                   level = 3; status = "SEVERE"

    # Draw overlay
    out = original.copy()
    if level > 0:
        red = original.copy()
        red[crack_mask == 255] = [0, 0, 255]
        out = cv2.addWeighted(original, 0.55, red, 0.55, 0)

    for (x1, y1, x2, y2), conf in zip(ai_boxes, ai_confs):
        cv2.rectangle(out, (x1,y1), (x2,y2), (255,0,200), 2)
        lbl = f"Crack {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1-lh-8), (x1+lw+6, y1), (255,0,200), -1)
        cv2.putText(out, lbl, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    banner = {0:(0,160,60), 1:(0,180,180), 2:(0,130,255), 3:(0,0,210)}
    cv2.rectangle(out, (0,0), (w_img,48), banner[level], -1)
    cv2.putText(out, f"{status}  |  Area: {total_area}px  |  AI: {len(ai_boxes)} crack(s)",
                (12,33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)

    tag = f"YOLOv8: {MODEL_NAME}" if AI_READY else "Edge Only"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(out, (w_img-tw-12, h_img-th-14), (w_img, h_img), (0,0,0), -1)
    cv2.putText(out, tag, (w_img-tw-6, h_img-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1, cv2.LINE_AA)

    return out, level, total_area, status, len(ai_boxes), coverage


def generate_frames():
    cam        = get_camera()
    prev_time  = time.time()
    frame_cnt  = 0
    while True:
        success, frame = cam.read()
        if not success:
            break
        annotated, level, area, status, num_boxes, coverage = analyze_frame_live(frame)
        frame_cnt += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps = round(frame_cnt / (now - prev_time), 1)
            with stats_lock:
                live_stats.update({'level':level,'area':area,'status':status,
                                   'ai_boxes':num_boxes,'coverage':str(coverage),'fps':fps})
            frame_cnt = 0; prev_time = now
        ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    from flask import Response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_stats')
def live_stats_endpoint():
    from flask import jsonify
    with stats_lock:
        return jsonify(live_stats)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    from flask import jsonify
    release_camera()
    return jsonify({'status': 'stopped'})



# ─────────────────────────────────────────────
# ANALYZE LIVE FRAME (Browser → Server → AI → Browser)
# ─────────────────────────────────────────────
@app.route('/analyze_live', methods=['POST'])
def analyze_live():
    from flask import jsonify
    import base64, re
    try:
        data       = request.get_json()
        image_data = data.get('image', '')
        # Decode base64 frame
        img_bytes  = base64.b64decode(re.sub(r'^data:image/.+;base64,', '', image_data))
        nparr      = np.frombuffer(img_bytes, np.uint8)
        frame      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400

        original = frame.copy()
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_img, w_img = frame.shape[:2]

        # Edge detection
        blur       = cv2.GaussianBlur(gray, (5,5), 0)
        edges      = cv2.Canny(blur, 35, 130)
        kernel     = np.ones((3,3), np.uint8)
        crack_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        edge_area  = int(np.sum(crack_mask > 0) / 4)
        coverage   = round((np.sum(crack_mask > 0) / (w_img * h_img)) * 100, 2)

        # YOLO AI
        ai_area  = 0
        ai_boxes = []
        ai_confs = []
        if AI_READY and model is not None:
            try:
                results = model(frame, conf=0.20, iou=0.45, imgsz=640, verbose=False)
                result  = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        ai_boxes.append((x1,y1,x2,y2))
                        ai_confs.append(conf)
                        ai_area += int((x2-x1)*(y2-y1)*conf*0.4)
            except Exception as e:
                print(f"Live AI error: {e}")

        # Hybrid score
        total_area = int(edge_area*0.45 + ai_area*1.55) if ai_boxes else int(edge_area*0.85)
        if   total_area < 50:   level = 0; status = "SAFE"
        elif total_area < 2500: level = 1; status = "MINOR"
        elif total_area < 8000: level = 2; status = "MEDIUM"
        else:                   level = 3; status = "SEVERE"

        # Draw overlay
        out = original.copy()
        if level > 0:
            red = original.copy()
            red[crack_mask == 255] = [0, 0, 255]
            out = cv2.addWeighted(original, 0.55, red, 0.55, 0)

        for (x1,y1,x2,y2), conf in zip(ai_boxes, ai_confs):
            cv2.rectangle(out, (x1,y1), (x2,y2), (255,0,200), 2)
            lbl = f"Crack {conf:.0%}"
            (lw,lh),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1,y1-lh-8), (x1+lw+6,y1), (255,0,200), -1)
            cv2.putText(out, lbl, (x1+3,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        banner = {0:(0,160,60), 1:(0,180,180), 2:(0,130,255), 3:(0,0,210)}
        cv2.rectangle(out, (0,0), (w_img,44), banner[level], -1)
        cv2.putText(out, f"{status}  |  Area: {total_area}px  |  AI: {len(ai_boxes)} crack(s)",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        tag = f"YOLOv8: {MODEL_NAME}" if AI_READY else "Edge Only"
        (tw,th),_ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (w_img-tw-12, h_img-th-14), (w_img, h_img), (0,0,0), -1)
        cv2.putText(out, tag, (w_img-tw-6, h_img-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1, cv2.LINE_AA)

        # Encode annotated frame back to base64
        _, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()

        return jsonify({
            'level':           level,
            'area':            total_area,
            'status':          status,
            'ai_boxes':        len(ai_boxes),
            'coverage':        str(coverage),
            'annotated_image': annotated_b64
        })
    except Exception as e:
        print(f"analyze_live error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    try:
        conn    = get_db()
        records = conn.execute(
            "SELECT id, image_path, crack_level, crack_area, timestamp FROM cracks ORDER BY timestamp DESC"
        ).fetchall()
        conn.close()
        # Convert to plain tuples so template works the same
        records = [tuple(r) for r in records]
        return render_template('history.html', records=records, db_status="Connected")
    except Exception as e:
        print(f"History error: {e}")
        return render_template('history.html', records=[], db_status=f"Error: {e}")


@app.route('/delete/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    try:
        conn = get_db()
        conn.execute("DELETE FROM cracks WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Delete error: {e}")
    return redirect(url_for('history'))


if __name__ == '__main__':
    import os; port = int(os.environ.get('PORT', 5000)); app.run(debug=False, host='0.0.0.0', port=port, threaded=True)