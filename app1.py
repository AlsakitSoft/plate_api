# -------------------------------------------------------------------
# المرحلة 1: استيراد المكتبات الضرورية
# -------------------------------------------------------------------
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import traceback  # مكتبة مفيدة لعرض الأخطاء التفصيلية

# -------------------------------------------------------------------
# المرحلة 2: تهيئة التطبيق وتحميل النماذج
# يتم هذا الجزء مرة واحدة فقط عند بدء تشغيل الخادم
# -------------------------------------------------------------------
app = Flask(__name__)
# ✅ تحديد مسارات المجلدات
UPLOADS_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'annotated'

print("--> جاري تحميل النماذج، يرجى الانتظار...")
try:
    # تأكد من أن اسم الملف صحيح وموجود في نفس المجلد
    model = YOLO("license_plate_detector.pt")

    # ✅ إضافة اللغة العربية إلى EasyOCR لتحسين الدقة للوحات السعودية
    ocr_reader = easyocr.Reader(['en', 'ar'], gpu=False)  # استخدم gpu=False للخوادم

    print("✅ تم تحميل النماذج بنجاح!")
except Exception as e:
    print(f"❌ حدث خطأ فادح أثناء تحميل النماذج: {e}")
    # إذا فشل تحميل النماذج، سنجعل المتغيرات فارغة لمنع تعطل التطبيق
    model = None
    ocr_reader = None


# -------------------------------------------------------------------
# المرحلة 3: تعريف "نقطة الوصول" (Endpoint)
# هذا هو الرابط الذي سيقوم تطبيق فلاتر بزيارته
# -------------------------------------------------------------------
@app.route('/recognize', methods=['POST'])
def recognize_plate_api():
    # التحقق أولاً من أن النماذج تم تحميلها بنجاح
    if model is None or ocr_reader is None:
        return jsonify({'success': False, 'error': 'النماذج غير محملة على الخادم'}), 500

    # التحقق من أن الطلب القادم من فلاتر يحتوي على ملف صورة
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'لم يتم إرسال ملف صورة'}), 400

    file = request.files['image']

    try:
        # قراءة بيانات الصورة من الطلب
        img_bytes = file.read()
        # تحويل البيانات إلى صيغة يمكن لـ OpenCV التعامل معها
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- هنا يبدأ منطق المعالجة المأخوذ من الكود الأصلي ---

        # 1. استخدام YOLO لتحديد مكان اللوحة
        results = model.predict(source=img, conf=0.4, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return jsonify({'success': False, 'error': 'لم يتم اكتشاف لوحة في الصورة'})

        # استهداف أول لوحة يتم العثور عليها
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # قص صورة اللوحة من الصورة الأصلية
        cropped_plate = img[y1:y2, x1:x2]

        # # 2. استخدام EasyOCR لقراءة النص من اللوحة المقصوصة
        # # تحسين الصورة لـ OCR (اختياري ولكنه مفيد)
        # gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        #
        # ocr_results = ocr_reader.readtext(gray_plate)
        #
        # if not ocr_results:
        #     return jsonify({'success': False, 'error': 'فشل التعرف الضوئي على الحروف'})
        #
        # # تجميع كل النصوص التي تم التعرف عليها بثقة معقولة
        # plate_number = " ".join([res[1] for res in ocr_results if res[2] > 0.3])
        #
        # # 3. إرجاع النتيجة الناجحة إلى تطبيق فلاتر
        # return jsonify({
        #     'success': True,
        #     'plate_text': plate_number.strip().upper()
        # })
        # 2. استخدام EasyOCR لقراءة النص من اللوحة المقصوصة
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

        # ✅ الخطوة 1: تحديد الحروف والأرقام التي نتوقعها فقط
        # هذا يقلل بشكل كبير من الأخطاء والضوضاء
       # allowed_chars = '١٢٣٤٥٦٧٨٩٠0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        allowed_chars = '١٢٣٤٥٦٧٨٩٠'

        # ✅ الخطوة 2: تشغيل OCR مع القائمة المسموح بها
        ocr_results = ocr_reader.readtext(gray_plate, allowlist=allowed_chars)

        if not ocr_results:
            return jsonify({'success': False, 'error': 'فشل التعرف الضوئي على الحروف'})

        # ✅ الخطوة 3: منطق تجميع أكثر ذكاءً
        # سنقوم بفرز النصوص التي تم العثور عليها من اليسار إلى اليمين (أو من الأعلى إلى الأسفل)
        # للحفاظ على الترتيب الصحيح.

        # ocr_results تحتوي على (bounding_box, text, confidence)
        # سنفرزها بناءً على إحداثي x للنقطة العلوية اليسرى من الصندوق
        ocr_results.sort(key=lambda res: res[0][0][0])

        # تجميع النصوص التي لها ثقة معقولة
        plate_components = [res[1] for res in ocr_results if res[2] > 0.3]

        # تجميع المكونات في سلسلة واحدة
        plate_number = " ".join(plate_components)

        # ✅ الخطوة 4: تنظيف نهائي (اختياري ولكنه مفيد)
        # إزالة أي مسافات متعددة واستبدالها بمسافة واحدة
        plate_number = " ".join(plate_number.split())

        # 3. إرجاع النتيجة الناجحة إلى تطبيق فلاتر
        return jsonify({
            'success': True,
            'plate_text': plate_number.strip().upper()
        })


    except Exception as e:
        # في حالة حدوث أي خطأ غير متوقع أثناء المعالجة
        traceback.print_exc()  # يطبع الخطأ الكامل في طرفية الخادم (مفيد جدًا للتصحيح)
        return jsonify({'success': False, 'error': f'حدث خطأ داخلي في الخادم: {str(e)}'}), 500

@app.route('/dashboard')
def dashboard():
    try:
        # قراءة أسماء الملفات من المجلدين
        #uploaded_images = sorted(os.listdir(UPLOADS_FOLDER), reverse=True)
       # annotated_images = sorted(os.listdir(ANNOTATED_FOLDER), reverse=True)

        # إنشاء كود HTML
        html = "<html><head><title>Dashboard</title><style>body{font-family:sans-serif; display:flex; gap:20px;} .col{flex:1;}</style></head><body>"
        html += "<div class='col'><h1>Original Images</h1>"
      #  for image in uploaded_images:
          #  html += f'<a href="/uploads/{image}" target="_blank"><img src="/uploads/{image}" width="300"></a><br><small>{image}</small><hr>'
        html += "</div>"

        html += "<div class='col'><h1>Annotated Images</h1>"
        # for image in annotated_images:
        #     html += f'<a href="/annotated/{image}" target="_blank"><img src="/annotated/{image}" width="300"></a><br><small>{image}</small><hr>'
        html += "</div>"

        html += "</body></html>"
        return html
    except Exception as e:
        return f"Error loading dashboard: {e}"
# -------------------------------------------------------------------
# المرحلة 4: تشغيل الخادم
# -------------------------------------------------------------------
if __name__ == '__main__':
    # host='0.0.0.0' يجعل الخادم متاحًا على شبكتك المحلية
    # port=5000 هو المنفذ الذي سيعمل عليه الخادم
    app.run(host='0.0.0.0', port=5000, debug=True)
