# -------------------------------------------------------------------
# المرحلة 1: استيراد المكتبات الضرورية
# -------------------------------------------------------------------
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import re
import os
from datetime import datetime
import traceback
from PIL import Image, ImageDraw, ImageFont

# -------------------------------------------------------------------
# المرحلة 2: تهيئة التطبيق
# -------------------------------------------------------------------
app = Flask(__name__)

# تحديد مسارات المجلدات
UPLOADS_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'annotated'
os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# تحديد مسار Tesseract-OCR
# ❗️ تنبيه: تأكد من أن هذا المسار صحيح على الخادم الذي ستشغل عليه الكود
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    print("تحذير: لم يتم العثور على Tesseract. قد تفشل عملية قراءة النص.")


# -------------------------------------------------------------------
# المرحلة 3: دمج كلاس معالجة الصور (YemeniPlateDetector)
# -------------------------------------------------------------------
class YemeniPlateDetector:
    def __init__(self, model_path="best.pt"):
        print("--> جاري تحميل نموذج YOLO، يرجى الانتظار...")
        try:
            self.model = YOLO(model_path)
            print("✅ تم تحميل نموذج YOLO بنجاح!")
        except Exception as e:
            print(f"❌ حدث خطأ فادح أثناء تحميل نموذج YOLO: {e}")
            self.model = None

    def enhance_plate_image(self, plate_img, plate_type_name="unknown"):
        """تحسين متقدم لصورة اللوحة مع معالجة خاصة حسب نوعها."""
        if plate_img is None or plate_img.size == 0: return None

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        denoised = cv2.fastNlMeansDenoising(contrast_enhanced, h=10, templateWindowSize=7, searchWindowSize=21)

        if plate_type_name == "خصوصي":
            processed = cv2.bitwise_not(denoised)
            blurred = cv2.GaussianBlur(processed, (3, 3), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif plate_type_name == "أجرة":
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        else:
            blurred = cv2.medianBlur(denoised, 3)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((1, 1), np.uint8)
        final = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return final

    def split_plate_regions(self, image, bottom_percent=0.4):
        """فصل الجزء السفلي من اللوحة الذي يحتوي على الأرقام."""
        if image is None or image.size == 0: return None
        height, width = image.shape[:2]
        start_y = int(height * (1 - bottom_percent))
        bottom = image[start_y:, :]
        return bottom

    def extract_text_from_plate(self, plate_img, plate_type_name="unknown"):
        """استخراج النص من صورة اللوحة الكاملة بعد تحسينها وفصلها."""
        if plate_img is None: return ""

        enhanced_img = self.enhance_plate_image(plate_img, plate_type_name)
        numbers_region = self.split_plate_regions(enhanced_img)
        if numbers_region is None or numbers_region.size == 0: return ""

        resized_region = cv2.resize(numbers_region, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        custom_config = r'--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(resized_region, config=custom_config).strip()

        cleaned_text = re.sub(r'[^0-9]', ' ', text)
        return " ".join(cleaned_text.split())

    def extract_numbers(self, text):
        """فصل النص إلى رقم المحافظة ورقم اللوحة."""
        numbers = re.findall(r'\d+', text)
        if not numbers: return "", ""
        if len(numbers) == 1: return "", numbers[0]
        return numbers[0], numbers[1]


# إنشاء نسخة واحدة من الكلاس عند بدء تشغيل الخادم
detector = YemeniPlateDetector()


# -------------------------------------------------------------------
# المرحلة 4: نقطة الوصول الرئيسية (API Endpoint)
# -------------------------------------------------------------------
@app.route('/recognize', methods=['POST'])
def recognize_plate_api():
    if detector.model is None:
        return jsonify({'success': False, 'error': 'نموذج YOLO غير محمل على الخادم'}), 500
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'لم يتم إرسال ملف صورة'}), 400

    file = request.files['image']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        annotated_img = img.copy()

        results = detector.model.predict(source=img, conf=0.4, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return jsonify({
                'success': True,
                'plates_found': 0,
                'results': [],
                'message': 'لم يتم اكتشاف أي لوحات في الصورة'
            })

        all_plates_data = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            CATEGORY_MAP = {'1': 'خصوصي', '2': 'أجرة', '3': 'نقل'}
            class_id = int(box.cls[0])
            class_name = detector.model.names[class_id]
            plate_type = CATEGORY_MAP.get(class_name, "غير معروف")

            cropped_plate = img[y1:y2, x1:x2]

            plate_text = detector.extract_text_from_plate(cropped_plate, plate_type)
            province_number, plate_number_digits = detector.extract_numbers(plate_text)

            if not province_number and plate_number_digits:
                plate_type = "مؤقت"

            display_text = f"{plate_type} - {plate_text if plate_text else 'READ_FAILED'}"
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated_img, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            plate_data = {
                'plate_type': plate_type,
                'full_text': plate_text,
                'province_number': province_number,
                'plate_number': plate_number_digits,
                'bounding_box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            }

            all_plates_data.append(plate_data)

        annotated_filename = f"annotated_{timestamp}.jpg"
        annotated_filepath = os.path.join(ANNOTATED_FOLDER, annotated_filename)
        cv2.imwrite(annotated_filepath, annotated_img)

        return jsonify({
            'success': True,
            'plates_found': len(all_plates_data),
            'results': all_plates_data,
            'annotated_image_url': f"/annotated/{annotated_filename}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'حدث خطأ داخلي في الخادم: {str(e)}'}), 500


# -------------------------------------------------------------------
# المرحلة 5: نقاط الوصول لعرض الصور ولوحة التحكم
# -------------------------------------------------------------------
@app.route('/dashboard')
def dashboard():
    try:
        def get_files_sorted_by_time(folder):
            files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return [os.path.basename(f) for f in files]

        annotated_images = get_files_sorted_by_time(ANNOTATED_FOLDER)
        html = f"""
        <html><head><title>Recognition Dashboard</title><meta http-equiv="refresh" content="10"><style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;background-color:#121212;color:#e0e0e0;margin:0;padding:20px;}}h1{{text-align:center;color:#fff;border-bottom:2px solid #333;padding-bottom:10px;}}
        .container{{display:grid;grid-template-columns:repeat(auto-fill,minmax(350px,1fr ));gap:20px;}}.card{{background-color:#1e1e1e;border:1px solid #333;border-radius:12px;overflow:hidden;box-shadow:0 4px 8px rgba(0,0,0,0.2);transition:transform .2s ease-in-out;}}
        .card:hover{{transform:translateY(-5px);}}.card img{{width:100%;height:auto;display:block;}}.card-content{{padding:15px;}}.card-content h3{{margin-top:0;color:#bb86fc;}}
        .card-content small{{color:#888;word-wrap:break-word;}}a{{text-decoration:none;color:inherit;}}</style></head><body><h1>Image Recognition Dashboard</h1><div class="container">
        """
        if not annotated_images:
            html += "<p style='text-align:center;'>No images processed yet. Send an image to /recognize.</p>"
        else:
            for image_name in annotated_images:
                html += f"""
                <div class="card"><a href="/annotated/{image_name}" target="_blank"><img src="/annotated/{image_name}" alt="Annotated Image"></a>
                <div class="card-content"><h3>Processed Image</h3><small>Filename: {image_name}</small></div></div>
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
    app.run(host='0.0.0.0', port=5000, debug=True)



