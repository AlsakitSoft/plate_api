# -------------------------------------------------------------------
# المرحلة 1: استيراد المكتبات الضرورية
# -------------------------------------------------------------------
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import traceback
import os  # ✅ مكتبة للتعامل مع الملفات والمجلدات
from datetime import datetime  # ✅ لاستخدام الوقت في تسمية الملفات

# -------------------------------------------------------------------
# المرحلة 2: تهيئة التطبيق والمجلدات
# -------------------------------------------------------------------
app = Flask(__name__)

# ✅ تحديد مسارات المجلدات
UPLOADS_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'annotated'
# ✅ التأكد من أن المجلدات موجودة
os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# ... (كود تحميل النماذج يبقى كما هو) ...
print("--> جاري تحميل النماذج، يرجى الانتظار...")
try:
    model = YOLO("license_plate_detector.pt")
    ocr_reader = easyocr.Reader(['en', 'ar'], gpu=False)
    print("✅ تم تحميل النماذج بنجاح!")
except Exception as e:
    print(f"❌ حدث خطأ فادح أثناء تحميل النماذج: {e}")
    model = None
    ocr_reader = None


# -------------------------------------------------------------------
# المرحلة 3: تعديل نقطة الوصول الرئيسية
# -------------------------------------------------------------------
@app.route('/recognize', methods=['POST'])
def recognize_plate_api():
    if model is None or ocr_reader is None:
        return jsonify({'success': False, 'error': 'النماذج غير محملة على الخادم'}), 500
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'لم يتم إرسال ملف صورة'}), 400

    file = request.files['image']

    # ✅ إنشاء اسم ملف فريد باستخدام الوقت الحالي
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_filename = f"original_{timestamp}.jpg"
    original_filepath = os.path.join(UPLOADS_FOLDER, original_filename)

    try:
        img_bytes = file.read()

        # ✅ حفظ الصورة الأصلية (اختياري)
        with open(original_filepath, 'wb') as f:
            f.write(img_bytes)

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- منطق المعالجة ---
        results = model.predict(source=img, conf=0.4, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return jsonify({'success': False, 'error': 'لم يتم اكتشاف لوحة في الصورة'})

        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_plate = img[y1:y2, x1:x2]
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

        allowed_chars = '١٢٣٤٥٦٧٨٩٠0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ocr_results = ocr_reader.readtext(gray_plate, allowlist=allowed_chars)

        if not ocr_results:
            plate_number = "OCR_FAILED"
        else:
            ocr_results.sort(key=lambda res: res[0][0][0])
            plate_number = " ".join([res[1] for res in ocr_results if res[2] > 0.3])
            plate_number = " ".join(plate_number.split())

        # ✅ --- رسم النتائج على الصورة ---
        annotated_img = img.copy()
        # رسم المربع حول اللوحة
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # كتابة النص فوق المربع
        # (ملاحظة: OpenCV لا يدعم العربية مباشرة، قد تظهر كـ ؟؟؟)
        # سنستخدم حيلة بسيطة لعرضها باستخدام PIL
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            # حاول استخدام خط يدعم العربية
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
        draw.text((x1, y1 - 50), plate_number.strip().upper(), font=font, fill=(255, 0, 0))
        annotated_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # ✅ حفظ الصورة المعلمة
        annotated_filename = f"annotated_{timestamp}_{plate_number.strip().upper()}.jpg"
        annotated_filepath = os.path.join(ANNOTATED_FOLDER, annotated_filename)
        cv2.imwrite(annotated_filepath, annotated_img)

        return jsonify({
            'success': True,
            'plate_text': plate_number.strip().upper()
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'حدث خطأ داخلي في الخادم: {str(e)}'}), 500


# -------------------------------------------------------------------
# ✅ المرحلة 4: إضافة نقاط وصول جديدة لعرض الصور
# -------------------------------------------------------------------

# واجهة HTML بسيطة لعرض الصور
# @app.route('/dashboard')
# def dashboard():
#     try:
#         # قراءة أسماء الملفات من المجلدين
#         uploaded_images = sorted(os.listdir(UPLOADS_FOLDER), reverse=True)
#         annotated_images = sorted(os.listdir(ANNOTATED_FOLDER), reverse=True)
#
#         # إنشاء كود HTML
#         html = "<html><head><title>Dashboard</title><style>body{font-family:sans-serif; display:flex; gap:20px;} .col{flex:1;}</style></head><body>"
#         html += "<div class='col'><h1>Original Images</h1>"
#         for image in uploaded_images:
#             html += f'<a href="/uploads/{image}" target="_blank"><img src="/uploads/{image}" width="300"></a><br><small>{image}</small><hr>'
#         html += "</div>"
#
#         html += "<div class='col'><h1>Annotated Images</h1>"
#         for image in annotated_images:
#             html += f'<a href="/annotated/{image}" target="_blank"><img src="/annotated/{image}" width="300"></a><br><small>{image}</small><hr>'
#         html += "</div>"
#
#         html += "</body></html>"
#         return html
#     except Exception as e:
#         return f"Error loading dashboard: {e}"




#http://192.168.173.96:5000/dashboard
#http://127.0.0.1:5000/dashboard
# واجهة HTML بسيطة لعرض الصور
@app.route('/dashboard')
def dashboard():
    try:
        # قراءة أسماء الملفات من المجلدين
        # نستخدم os.path.getmtime للحصول على وقت تعديل الملف للفرز الدقيق
        def get_files_sorted_by_time(folder):
            files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return [os.path.basename(f) for f in files]

        annotated_images = get_files_sorted_by_time(ANNOTATED_FOLDER)

        # ✅ --- بداية كود الـ HTML والـ CSS المحسّن ---

        # استخدام f-string متعدد الأسطر لتسهيل القراءة
        html = f"""
        <html>
        <head>
            <title>Recognition Dashboard</title>
            <meta http-equiv="refresh" content="10"> 
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    background-color: #121212;
                    color: #e0e0e0;
                    margin: 0;
                    padding: 20px;
                }}
                h1 {{
                    text-align: center;
                    color: #ffffff;
                    border-bottom: 2px solid #333;
                    padding-bottom: 10px;
                }}
                .container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr ));
                    gap: 20px;
                }}
                .card {{
                    background-color: #1e1e1e;
                    border: 1px solid #333;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    transition: transform 0.2s ease-in-out;
                }}
                .card:hover {{
                    transform: translateY(-5px);
                }}
                .card img {{
                    width: 100%;
                    height: auto;
                    display: block;
                }}
                .card-content {{
                    padding: 15px;
                }}
                .card-content h3 {{
                    margin-top: 0;
                    color: #bb86fc; /* لون بنفسجي فاتح */
                }}
                .card-content small {{
                    color: #888;
                    word-wrap: break-word;
                }}
                a {{
                    text-decoration: none;
                    color: inherit;
                }}
            </style>
        </head>
        <body>
            <h1>Image Recognition Dashboard</h1>
            <div class="container">
        """

        # ✅ عرض الصور المعالجة فقط في بطاقات منظمة
        # لم نعد بحاجة لعرض الصور الأصلية لأنها موجودة ضمن المعالجة
        if not annotated_images:
            html += "<p style='text-align:center;'>No images processed yet. Send an image from the app.</p>"
        else:
            for image_name in annotated_images:
                # استخراج رقم اللوحة من اسم الملف
                try:
                    # الاسم يكون annotated_timestamp_PLATENUMBER.jpg
                    plate_number = image_name.split('_')[2].split('.jpg')[0]
                except IndexError:
                    plate_number = "N/A"

                html += f"""
                <div class="card">
                    <a href="/annotated/{image_name}" target="_blank">
                        <img src="/annotated/{image_name}" alt="Annotated Image">
                    </a>
                    <div class="card-content">
                        <h3>Recognized: {plate_number}</h3>
                        <small>Filename: {image_name}</small>
                    </div>
                </div>
                """

        html += """
            </div>
        </body>
        </html>
        """
        # ✅ --- نهاية كود الـ HTML والـ CSS المحسّن ---

        return html
    except Exception as e:
        traceback.print_exc()
        return f"Error loading dashboard: {e}"


# نقطة وصول لخدمة الصور من مجلد 'uploads'
@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(UPLOADS_FOLDER, filename)


# نقطة وصول لخدمة الصور من مجلد 'annotated'
@app.route('/annotated/<filename>')
def send_annotated_file(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)


# -------------------------------------------------------------------
# المرحلة 5: تشغيل الخادم
# -------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
