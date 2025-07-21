

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
# ❗️ تأكد من أن هذا المسار صحيح على الخادم أو قم بتعيينه في متغيرات البيئة
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    print("تحذير: لم يتم العثور على Tesseract. قد تفشل عملية قراءة النص.")


# -------------------------------------------------------------------
# المرحلة 3: كلاس معالجة الصور الموحّد والمحسّن
# -------------------------------------------------------------------
class YemeniPlateDetector:
    def __init__(self, model_path="best.pt"):
        print("--> جاري تحميل نموذج YOLO، يرجى الانتظار...")
        try:
            self.model = YOLO(model_path)
            # ربط الأكواد بأنواع اللوحات (تأكد من مطابقتها لملف .yaml الخاص بالنموذج)
            self.CATEGORY_MAP = self.model.names
            # مثال إذا كانت الأسماء في النموذج هي '0', '1', '2'
            # self.CATEGORY_MAP = {'0': 'خصوصي', '1': 'أجرة', '2': 'نقل'}
            print(f"✅ تم تحميل نموذج YOLO بنجاح! الفئات المكتشفة: {self.CATEGORY_MAP}")
        except Exception as e:
            print(f"❌ حدث خطأ فادح أثناء تحميل نموذج YOLO: {e}")
            self.model = None

    def enhance_plate_image(self, plate_img, plate_type="unknown"):
        """تحسين متقدم لصورة اللوحة مع معالجة خاصة حسب نوعها."""
        if plate_img is None or plate_img.size == 0: return None
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # 1. CLAHE لتحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        # 2. إزالة الضوضاء
        denoised = cv2.fastNlMeansDenoising(contrast, h=10, templateWindowSize=7, searchWindowSize=21)

        # 3. معالجة خاصة حسب نوع اللوحة (مثل اللوحات الزرقاء "خصوصي")
        if plate_type.lower() == "خصوصي":
            # عكس الألوان قد يساعد في قراءة النص الأبيض على خلفية داكنة
            processed = cv2.bitwise_not(denoised)
            blurred = cv2.GaussianBlur(processed, (3, 3), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # الأنواع الأخرى
            blurred = cv2.medianBlur(denoised, 3)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. عمليات مورفولوجية لتنظيف الحواف
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return cleaned

    def split_plate_regions(self, image, bottom_percent=0.4):
        """فصل الجزء السفلي من اللوحة الذي يحتوي على الأرقام."""
        if image is None or image.size == 0: return None
        h, w = image.shape[:2]
        start_y = int(h * (1 - bottom_percent))
        region = image[start_y:, :]
        return region

    def extract_text_from_plate(self, plate_img, plate_type="unknown"):
        """استخراج النص من صورة اللوحة بعد تحسينها وفصلها."""
        if plate_img is None: return ""

        enhanced = self.enhance_plate_image(plate_img, plate_type)
        region = self.split_plate_regions(enhanced)

        if region is None or region.size == 0: return ""

        # تكبير الصورة لتحسين الدقة
        region = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # استخدام Tesseract مع تحديد الأرقام فقط
       # config = r'--oem 3 --psm 6 -l eng --tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata" -c tessedit_char_whitelist=0123456789'
        config = r'--oem 3 --psm 6 -l eng'
        raw_text = pytesseract.image_to_string(region, config=config).strip()

        # تنظيف النص من أي رموز غير رقمية
        cleaned_text = re.sub(r'[^0-9]', '', raw_text)
        return cleaned_text

    def extract_numbers(self, text):
        """فصل النص إلى رقم المحافظة والرقم الرئيسي."""
        # هذه الدالة تحتاج إلى منطق مخصص حسب تنسيق الأرقام في اليمن
        # افتراض: رقم المحافظة (1-2 أرقام)، الباقي هو رقم اللوحة
        if len(text) <= 2:
            return text, ""  # قد يكون رقم محافظة فقط
        if len(text) > 2 and len(text) < 5:  # على الأغلب رقم لوحة بدون محافظة
            return "", text

        # افتراض أن أول رقمين هما للمحافظة إذا كان الطول مناسباً
        province = text[:2]
        plate_num = text[2:]
        # يمكنك تحسين هذا المنطق بناءً على قواعد أرقام اللوحات
        return province, plate_num

    def analyze_image(self, img):
        """الكشف والتحليل الكامل للصورة المدخلة."""
        image = img.copy()
        results = self.model.predict(source=image, conf=0.4, verbose=False)
        annotated_img = image.copy()
        plates_data = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image[y1:y2, x1:x2]

                # تحديد نوع اللوحة من النموذج
                class_id = int(box.cls[0])
                plate_type = self.CATEGORY_MAP.get(class_id, "غير معروف")

                # استخراج النص بناءً على النوع
                text = self.extract_text_from_plate(crop, plate_type)
                prov, num = self.extract_numbers(text)

                # تحديث النوع إذا كانت مؤقت (لا يوجد رقم محافظة)
                final_plate_type = "مؤقت" if not prov and num else plate_type

                label = f"{final_plate_type}: {prov}-{num}" if prov or num else f"{final_plate_type}: READ_FAILED"

                # رسم النتائج على الصورة
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                plates_data.append({
                    'bounding_box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'plate_type': final_plate_type,
                    'province_number': prov,
                    'plate_number': num,
                    'confidence': float(box.conf[0])
                })
        return plates_data, annotated_img


# إنشاء كائن للكاشف
_detector = YemeniPlateDetector()


# -------------------------------------------------------------------
# المرحلة 4: نقطة الوصول الرئيسية (API Endpoint)
# -------------------------------------------------------------------
@app.route('/recognize', methods=['POST'])
def recognize_plate_api():
    if _detector.model is None:
        return jsonify({'success': False, 'error': 'نموذج YOLO غير محمل'}), 500
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'لم تقدم صورة'}), 400

    try:
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        plates, annotated_image = _detector.analyze_image(img)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"annotated_{timestamp}.jpg"
        output_path = os.path.join(ANNOTATED_FOLDER, filename)
        cv2.imwrite(output_path, annotated_image)

        return jsonify({
            'success': True,
            'plates_found': len(plates),
            'plates': plates,
            'annotated_image_url': f"/annotated/{filename}"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# -------------------------------------------------------------------
# المرحلة 5: خدمتي الصور ولوحة التحكم
# -------------------------------------------------------------------
@app.route('/annotated/<filename>')
def get_annotated(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)


@app.route('/dashboard')
def dashboard():
    try:
        files = sorted(
            os.listdir(ANNOTATED_FOLDER),
            key=lambda f: os.path.getmtime(os.path.join(ANNOTATED_FOLDER, f)),
            reverse=True
        )
        files_to_show = files[:50]  # عرض أحدث 50 صورة فقط

        html = '''
        <html><head><title>Dashboard</title><meta http-equiv="refresh" content="10">
        <style>
            body { font-family: sans-serif; } .container { display: flex; flex-wrap: wrap; gap: 20px; }
            .item { border: 1px solid #ccc; padding: 10px; text-align: center; } img { max-width: 400px; }
        </style></head><body><h1>Annotated Images</h1><div class="container">
        '''
        for f in files_to_show:
            html += f'<div class="item"><img src="/annotated/{f}"><p>{f}</p></div>'
        html += '</div></body></html>'
        return html
    except Exception as e:
        return f"Error loading dashboard: {e}"


# -------------------------------------------------------------------
# المرحلة 6: تشغيل الخادم
# -------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
