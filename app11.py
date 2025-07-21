# -------------------------------------------------------------------
# المرحلة 1: استيراد المكتبات الضرورية
# -------------------------------------------------------------------
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract  # ✅ استبدال easyocr بـ pytesseract
import re
import os
from datetime import datetime
import traceback
from PIL import Image, ImageDraw, ImageFont

# -------------------------------------------------------------------
# المرحلة 2: تهيئة التطبيق والنماذج
# -------------------------------------------------------------------
app = Flask(__name__)

# ✅ تحديد مسارات المجلدات
UPLOADS_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'annotated'
os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# ✅ **مهم جداً:** تحديد مسار Tesseract-OCR
# تأكد من أن هذا المسار صحيح على جهازك أو الخادم
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # لنظام ويندوز
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # لنظام لينكس
except Exception:
    print("تحذير: لم يتم العثور على Tesseract في المسار الافتراضي. قد تفشل عملية قراءة النص.")

# تحميل نموذج YOLO
print("--> جاري تحميل نموذج YOLO، يرجى الانتظار...")
try:
    model = YOLO("best.pt")
    print("✅ تم تحميل نموذج YOLO بنجاح!")
except Exception as e:
    print(f"❌ حدث خطأ فادح أثناء تحميل نموذج YOLO: {e}")
    model = None


# -------------------------------------------------------------------
# المرحلة 3: إضافة دوال التحسين وقراءة النص (من كود Tkinter)
# -------------------------------------------------------------------

def enhance_plate_image(plate_img):
    """تحسين صورة اللوحة لزيادة دقة القراءة."""
    if plate_img is None or plate_img.size == 0:
        return None

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # 1. زيادة التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # 2. إزالة الضوضاء
    denoised = cv2.fastNlMeansDenoising(contrast_enhanced, h=10, templateWindowSize=7, searchWindowSize=21)

    # 3. تطبيق Thresholding
    blurred = cv2.medianBlur(denoised, 3)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. تكبير الصورة لتحسين الدقة
    final_image = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return final_image


def extract_text_from_plate(enhanced_img):
    """استخراج النص من الصورة المحسنة باستخدام Pytesseract."""
    if enhanced_img is None:
        return ""

    # إعدادات Tesseract لقراءة الأرقام الإنجليزية
    custom_config = r'--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(enhanced_img, config=custom_config).strip()

    # تنظيف النص: إزالة أي شيء ليس رقماً واستبداله بمسافة
    cleaned_text = re.sub(r'[^0-9]', ' ', text)
    # إزالة المسافات المتعددة
    return " ".join(cleaned_text.split())


def extract_numbers(text):
    """فصل النص إلى رقم المحافظة ورقم اللوحة."""
    numbers = re.findall(r'\d+', text)
    if not numbers:
        return "", ""
    if len(numbers) == 1:
        return "", numbers[0]
    return numbers[0], numbers[1]


# -------------------------------------------------------------------
# المرحلة 4: تعديل نقطة الوصول الرئيسية /recognize
# -------------------------------------------------------------------
@app.route('/recognize', methods=['POST'])
def recognize_plate_api():
    if model is None:
        return jsonify({'success': False, 'error': 'نموذج YOLO غير محمل على الخادم'}), 500
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'لم يتم إرسال ملف صورة'}), 400

    file = request.files['image']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_filename = f"original_{timestamp}.jpg"
    original_filepath = os.path.join(UPLOADS_FOLDER, original_filename)

    try:
        img_bytes = file.read()
        with open(original_filepath, 'wb') as f:
            f.write(img_bytes)

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- منطق المعالجة الجديد ---
        results = model.predict(source=img, conf=0.4, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return jsonify({'success': False, 'error': 'لم يتم اكتشاف لوحة في الصورة'})

        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 1. قص اللوحة من الصورة الأصلية
        cropped_plate = img[y1:y2, x1:x2]

        # 2. تحسين صورة اللوحة
        enhanced_plate = enhance_plate_image(cropped_plate)

        # 3. استخراج النص باستخدام Pytesseract
        plate_text = extract_text_from_plate(enhanced_plate)

        # 4. فصل الأرقام
        province_number, plate_number_digits = extract_numbers(plate_text)

        # تحديد النص النهائي الذي سيعرض على الصورة
        display_text = plate_text if plate_text else "READ_FAILED"

        # --- رسم النتائج على الصورة ---
        annotated_img = img.copy()
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        pil_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
        draw.text((x1, y1 - 50), display_text, font=font, fill=(255, 0, 0))
        annotated_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # حفظ الصورة المعلمة
        annotated_filename = f"annotated_{timestamp}_{display_text.replace(' ', '_')}.jpg"
        annotated_filepath = os.path.join(ANNOTATED_FOLDER, annotated_filename)
        cv2.imwrite(annotated_filepath, annotated_img)

        # إرجاع استجابة JSON مفصلة
        return jsonify({
            'success': True,
            'full_text': plate_text,
            'province_number': province_number,
            'plate_number': plate_number_digits,
            'annotated_image_url': f"/annotated/{annotated_filename}"  # رابط لعرض الصورة
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'حدث خطأ داخلي في الخادم: {str(e)}'}), 500


# -------------------------------------------------------------------
# المرحلة 5: نقاط الوصول لعرض الصور ولوحة التحكم (تبقى كما هي)
# -------------------------------------------------------------------
@app.route('/dashboard')
def dashboard():
    # ... (كود لوحة التحكم لم يتغير) ...
    try:
        def get_files_sorted_by_time(folder):
            files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return [os.path.basename(f) for f in files]

        annotated_images = get_files_sorted_by_time(ANNOTATED_FOLDER)
        html = f"""
        <html>
        <head>
            <title>Recognition Dashboard</title>
            <meta http-equiv="refresh" content="10"> 
            <style>
                body {{ font-family: sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
                h1 {{ text-align: center; color: #ffffff; }}
                .container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr )); gap: 20px; }}
                .card {{ background-color: #1e1e1e; border: 1px solid #333; border-radius: 12px; overflow: hidden; }}
                .card:hover {{ transform: translateY(-5px); }}
                .card img {{ width: 100%; height: auto; }}
                .card-content {{ padding: 15px; }}
                .card-content h3 {{ margin-top: 0; color: #bb86fc; }}
                .card-content small {{ color: #888; word-wrap: break-word; }}
                a {{ text-decoration: none; color: inherit; }}
            </style>
        </head>
        <body>
            <h1>Image Recognition Dashboard</h1>
            <div class="container">
        """
        if not annotated_images:
            html += "<p style='text-align:center;'>No images processed yet.</p>"
        else:
            for image_name in annotated_images:
                try:
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
        html += "</div></body></html>"
        return html
    except Exception as e:
        traceback.print_exc()
        return f"Error loading dashboard: {e}"


@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(UPLOADS_FOLDER, filename)


@app.route('/annotated/<filename>')
def send_annotated_file(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)


# -------------------------------------------------------------------
# المرحلة 6: تشغيل الخادم
# -------------------------------------------------------------------
if __name__ == '__main__':
    # استخدم '0.0.0.0' لجعله متاحاً على شبكتك المحلية
    app.run(host='0.0.0.0', port=5000, debug=True)




