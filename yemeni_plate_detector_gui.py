

from datetime import datetime
from ultralytics import YOLO 
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pytesseract
import re
import os
import logging
#import arabic_reshaper
from bidi.algorithm import get_display

# تكوين التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure Tesseract is in PATH or specify its path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # For Linux

class YemeniPlateDetector:
    def __init__(self):
        self.model = YOLO("best.pt")  # نموذجك المدرب لاكتشاف لوحات يمنية

    # deepseek
    def enhance_plate_image(self, plate_img, plate_type="unknown"):
        """تحسين متقدم لصورة اللوحة اليمنية مع معالجة خاصة للتشوهات"""
        if plate_img is None or plate_img.size == 0:
            return None
        
        # التحويل إلى تدرج الرمادي إذا كانت صورة ملونة
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # 1. معالجة التباين (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        
        # 2. إزالة الضوضاء (Non-local Means Denoising)
        denoised = cv2.fastNlMeansDenoising(contrast_enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 3. معالجة خاصة حسب نوع اللوحة
        if plate_type == "private":  # اللوحات الزرقاء (خصوصي)
            processed = cv2.bitwise_not(denoised)  # عكس الألوان للنص الأبيض
            blurred = cv2.GaussianBlur(processed, (3,3), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif plate_type == "taxi":   # لوحات الأجرة
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((2,2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        else:  # الأنواع الأخرى
            blurred = cv2.medianBlur(denoised, 3)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. تحسين الحواف (Morphological Operations)
        kernel = np.ones((1,1), np.uint8)
        final = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return final
        
    def split_plate_regions(self, image, bottom_percent=0.4):
        """
        تقسيم اللوحة مع تحديد نسبة الجزء السفلي
        :param image: صورة اللوحة الكاملة
        :param bottom_percent: النسبة المئوية للجزء السفلي (0.4 تعني 40%)
        :return: الجزء السفلي من الصورة
        """
        height, width = image.shape[:2]
        
        # التحقق من أن النسبة بين 0.1 و 0.9
        bottom_percent = max(0.1, min(0.9, bottom_percent))
        
        start_y = int(height * (1 - bottom_percent))
        bottom = image[start_y:, :]
        
        # (اختياري) اقتصاص الأجزاء البيضاء الزائدة
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY) if len(bottom.shape) == 3 else bottom
        _, thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            bottom = bottom[y:y+h, x:x+w]
        
        return bottom

    def extract_text_from_plate(self, enhanced_img):
        """استخراج النص من اللوحة باستخدام الأرقام الإنجليزية فقط"""
        if enhanced_img is None:
            return "غير قابل للقراءة"

        enhanced_img = self.split_plate_regions(enhanced_img)
        # تكبير الصورة لتحسين دقة التعرف
        enhanced_img = cv2.resize(enhanced_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # استخدام Tesseract مع اللغة الإنجليزية فقط (أرقام إنجليزية فقط)
        # custom_config = r'--oem 3 --psm 6 -l ara+eng'
        # custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
        
        custom_config = r'--oem 3 --psm 6 -l eng'

        text = pytesseract.image_to_string(enhanced_img, config=custom_config).strip()

        # تنظيف النص: إزالة كل شيء غير أرقام أو مسافات
        text = re.sub(r'[^0-9]', ' ', text)

        # return text
        return text if text.strip() else "غير قابل للقراءة"

    def extract_numbers_before_space(input_text):
        """
        تستخرج الأرقام التي تسبق أول مسافة في النص
        
        :param input_text: النص المدخل (مثال: "1  2222" أو "22   66666")
        :return: الأرقام قبل أول مسافة أو النص كاملاً إذا لم توجد مسافات
        """
        # البحث عن أول تسلسل أرقام يسبق أول مسافة
        match = re.search(r'^(\d+)\s', input_text)
        
        if match:
            return match.group(1)  # إرجاع الأرقام قبل المسافة الأولى
        else:
            # إذا لم توجد مسافات، إرجاع كل الأرقام الموجودة (إن وجدت)
            return re.sub(r'[^\d]', '', input_text) or ''

    def extract_numbers(self, text):
        """
        تستخرج الرقمين قبل وبعد المسافة وتعيدهم كقيم منفصلة.
        - إذا كان هناك رقم واحد فقط، يعتبر هو الأساسي والرقم الآخر فارغ.
        - إذا وُجد رقمان أو أكثر، يتم أخذ أول رقمين فقط.
        """
        # استخراج كل الأرقام
        numbers = re.findall(r'\d+', text)

        if len(numbers) == 0:
            # لا يوجد أي رقم
            return ("", "")
        elif len(numbers) == 1:
            # يوجد رقم واحد فقط بعد المسافات
            return ("", numbers[0])
        else:
            # إرجاع أول رقمين
            return (numbers[0], numbers[1])


    def analyze_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            results = self.model(img)
            boxes = results[0].boxes      # كل الكائنات المكتشفة
            CATEGORY_MAP = {'1': 'خصوصي', '2': 'أجرة', '3': 'نقل'}
            plates = []


            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = img[y1:y2, x1:x2]
                enhanced_img = self.enhance_plate_image(plate_img)

                class_id = int(box.cls[0])  # المعرف الرقمي للفئة، مثل 0 أو 1 أو 2
                class_name = results[0].names[class_id]  # مثل '1' أو '2' أو '3'
                plate_type = CATEGORY_MAP.get(class_name, "غير معروف")

                text = self.extract_text_from_plate(enhanced_img)

                province_number , plate_number_digits =self.extract_numbers(text)
                # تحديث القيمة في حالة كانت مؤقت تعز                
                if province_number is None or province_number == "":
                    plate_type = "مؤقت"
                    class_name = 4 # الرقم الخاص معرف نوع السيارة

                confidence = "غير معروف"  # أو حسب نتائجك

                plates.append({
                    "plate_image_data": plate_img,
                    "enhanced_image_data": enhanced_img,
                    "text": text,
                    "plate_number_digits": plate_number_digits,
                    "province_number": province_number,
                    "type":  plate_type ,  # مثال فقط
                    "type_id":  class_name ,  # مثال فقط
                    "confidence": confidence,
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "plate_number": len(plates) + 1
                })

            return {
                "original_image_data": img,
                "results": plates,
                "plates_found": len(plates)
            }

        except Exception as e:
            return {"error": str(e)}


class PlateDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("كاشف اللوحات اليمنية")
        master.geometry("1000x800")
        master.resizable(True, True)

        self.detector = YemeniPlateDetector()

        # إعداد واجهة المستخدم
        self.setup_ui()

    def setup_ui(self):
        """تهيئة واجهة المستخدم"""
        # Main frame
        self.main_frame = tk.Frame(self.master, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        self.title_label = tk.Label(
            self.main_frame, 
            text="كاشف اللوحات اليمنية", 
            font=("Arial", 24, "bold"), 
            fg="#0056b3"
        )
        self.title_label.pack(pady=10)

        # Upload button
        self.upload_button = tk.Button(
            self.main_frame, 
            text="رفع صورة", 
            command=self.upload_image, 
            font=("Arial", 14), 
            bg="#007bff", 
            fg="white", 
            relief=tk.RAISED, 
            bd=3
        )
        self.upload_button.pack(pady=10)

        # Status/Error message
        self.status_label = tk.Label(
            self.main_frame, 
            text="", 
            font=("Arial", 12), 
            fg="red"
        )
        self.status_label.pack(pady=5)

        # Results display area
        self.results_frame = tk.Frame(
            self.main_frame, 
            bd=2, 
            relief=tk.GROOVE, 
            padx=10, 
            pady=10
        )
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.results_canvas = tk.Canvas(self.results_frame)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.results_scrollbar = tk.Scrollbar(
            self.results_frame, 
            orient="vertical", 
            command=self.results_canvas.yview
        )
        self.results_scrollbar.pack(side=tk.RIGHT, fill="y")

        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)
        self.results_canvas.bind('<Configure>', lambda e: self.results_canvas.configure(
            scrollregion=self.results_canvas.bbox("all")
        ))

        self.inner_results_frame = tk.Frame(self.results_canvas)
        self.results_canvas.create_window(
            (0, 0), 
            window=self.inner_results_frame, 
            anchor="nw"
        )

    def upload_image(self):
        """رفع صورة وتحليلها"""
        file_path = filedialog.askopenfilename(
            title="اختر صورة",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            self.status_label.config(text="جاري التحليل...", fg="blue")
            self.master.update_idletasks()
            self.analyze_and_display(file_path)

    def analyze_and_display(self, image_path):
        """تحليل الصورة وعرض النتائج"""
        for widget in self.inner_results_frame.winfo_children():
            widget.destroy()

        results = self.detector.analyze_image(image_path)

        if "error" in results:
            self.status_label.config(text=f"خطأ: {results['error']}", fg="red")
            return

        self.status_label.config(text="تم التحليل بنجاح!", fg="green")

        # عرض الصورة الأصلية
        original_img_tk = self.convert_cv2_to_tk(
            results['original_image_data'], 
            max_size=(600, 400)
        )
        original_img_label = tk.Label(
            self.inner_results_frame, 
            image=original_img_tk
        )
        original_img_label.image = original_img_tk
        original_img_label.pack(pady=10)
        
        tk.Label(
            self.inner_results_frame, 
            text="الصورة الأصلية", 
            font=("Arial", 16, "bold")
        ).pack()

        if results['plates_found'] > 0:
            tk.Label(
                self.inner_results_frame, 
                text="اللوحات المكتشفة:", 
                font=("Arial", 16, "bold")
            ).pack(pady=10)
            
            for plate in results['results']:
                plate_frame = tk.LabelFrame(
                    self.inner_results_frame, 
                    text=f"اللوحة رقم {plate['plate_number']}", 
                    font=("Arial", 14, "bold"), 
                    padx=10, 
                    pady=10
                )
                plate_frame.pack(pady=5, fill=tk.X, expand=True)

                # صورة اللوحة
                plate_img_tk = self.convert_cv2_to_tk(
                    plate['plate_image_data'], 
                    max_size=(200, 100)
                )
                plate_img_label = tk.Label(
                    plate_frame, 
                    image=plate_img_tk
                )
                plate_img_label.image = plate_img_tk
                plate_img_label.pack(side=tk.RIGHT, padx=5)

                # الصورة المحسنة (إذا كانت متاحة)
                if plate['enhanced_image_data'] is not None:
                    enhanced_img_tk = self.convert_cv2_to_tk(
                        plate['enhanced_image_data'], 
                        max_size=(200, 100)
                    )
                    enhanced_img_label = tk.Label(
                        plate_frame, 
                        image=enhanced_img_tk
                    )
                    enhanced_img_label.image = enhanced_img_tk
                    enhanced_img_label.pack(side=tk.RIGHT, padx=5)

                # تفاصيل اللوحة
                details_text = f"""
                النص: {plate['text']}
                رقم المحافظة: {plate['province_number']}
                رقم اللوحة: {plate['plate_number_digits']}
                النوع: {plate['type']}
                الثقة: {plate['confidence']}
                الإحداثيات (x, y, w, h): ({plate['bbox'][0]}, {plate['bbox'][1]}, {plate['bbox'][2]}, {plate['bbox'][3]})
                """
                details_label = tk.Label(
                    plate_frame, 
                    text=details_text, 
                    justify=tk.RIGHT, 
                    font=("Arial", 12)
                )
                details_label.pack(side=tk.RIGHT, padx=5)
        else:
            tk.Label(
                self.inner_results_frame, 
                text="لم يتم العثور على لوحات في الصورة.", 
                font=("Arial", 14)
            ).pack(pady=10)
        
        self.results_canvas.update_idletasks()
        self.results_canvas.config(scrollregion=self.results_canvas.bbox("all"))

    def convert_cv2_to_tk(self, cv2_img, max_size=(300, 300)):
        """تحويل صورة OpenCV إلى تنسيق متوافق مع Tkinter"""
        if cv2_img is None:
            return None
        
        h, w = cv2_img.shape[:2]
        if h > max_size[1] or w > max_size[0]:
            scaling_factor = min(max_size[0] / w, max_size[1] / h)
            new_w = int(w * scaling_factor)
            new_h = int(h * scaling_factor)
            cv2_img = cv2.resize(
                cv2_img, 
                (new_w, new_h), 
                interpolation=cv2.INTER_AREA
            )

        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        return img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = PlateDetectorApp(root)
    root.mainloop()



#### #################################################################

# from ultralytics import YOLO
# import cv2

# # تحميل النموذج
# model = YOLO('best.pt')

# # تحميل الصورة
# image = cv2.imread('images (2).jpeg')

# # إجراء الكشف
# results = model(image)[0]

# # ربط الأكواد بأنواع اللوحات
# CATEGORY_MAP = {'1': 'خصوصي', '2': 'أجرة', '3': 'نقل'}

# # استخراج النتائج
# for box in results.boxes:
#     class_id = int(box.cls[0])
#     class_name = results.names[class_id]  # هذا يكون '1' أو '2' أو '3'
#     plate_type = CATEGORY_MAP.get(class_name, "غير معروف")

#     # استخراج الإحداثيات
#     x1, y1, x2, y2 = map(int, box.xyxy[0])

#     print(f"تم كشف لوحة من نوع: {plate_type}")
#     print(f"الإحداثيات: ({x1}, {y1}), ({x2}, {y2})")

#     # رسم المستطيل على الصورة
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
#     cv2.putText(image, plate_type, (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

# # عرض الصورة
# cv2.imshow("Detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


