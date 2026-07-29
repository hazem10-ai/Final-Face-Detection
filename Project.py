import cv2
import dlib
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

class FaceDetectionBenchmarker:
    def __init__(self, dataset_path, output_path="Full_Project_Results"):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.stats = {}

        # 1. إعداد المجلدات لكل خوارزمية (سيتم حفظ 914 صورة لكل نوع)
        for algo in ["Haar_Cascade", "HOG_SVM", "Deep_Learning_DNN"]:
            os.makedirs(os.path.join(self.output_path, algo), exist_ok=True)
            self.stats[algo] = {"time": [], "True_Positives": 0, "False_Positives": 0, "False_Negatives": 0}

        # 2. تحميل الموديلات
        self.haar_net = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.hog_detector = dlib.get_frontal_face_detector()
        self.dnn_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")

    def _detect_haar(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.haar_net.detectMultiScale(gray, 1.1, 4)

    def _detect_hog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.hog_detector(gray, 1)
        return [(r.left(), r.top(), r.width(), r.height()) for r in rects]

    def _detect_dnn(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] > 0.5:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x2, y2) = box.astype("int")
                faces.append((x, y, x2-x, y2-y))
        return faces

    def run_full_benchmark(self, ground_truth_dict=None):
        """ معالجة وحفظ كافة الصور في الداتا سيت """
        print(f"🚀 Starting Comprehensive Benchmark on {len(self.image_files)} images...")

        for idx, filename in enumerate(self.image_files):
            img_path = os.path.join(self.dataset_path, filename)
            img = cv2.imread(img_path)
            if img is None: continue

            # تحديد عدد الوجوه الحقيقي (يفترض 1 إذا لم يحدد)
            actual_faces = ground_truth_dict.get(filename, 1) if ground_truth_dict else 1

            algorithms = [
                ("Haar_Cascade", self._detect_haar),
                ("HOG_SVM", self._detect_hog),
                ("Deep_Learning_DNN", self._detect_dnn)
            ]

            for name, method in algorithms:
                start_t = time.perf_counter()
                faces = method(img)
                end_t = time.perf_counter()

                self.stats[name]["time"].append((end_t - start_t) * 1000)
                
                det_count = len(faces)
                self.stats[name]["True_Positives"] += min(det_count, actual_faces)
                self.stats[name]["False_Positives"] += max(0, det_count - actual_faces)
                self.stats[name]["False_Negatives"] += max(0, actual_faces - det_count)

                # حفظ الصورة المكتشفة بداخل المجلد الخاص بها
                canvas = img.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(canvas, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(self.output_path, name, filename), canvas)

            if idx % 100 == 0:
                print(f"✅ Processed {idx}/{len(self.image_files)} images and saved all variants...")

    def generate_detailed_report(self):
        """ طباعة النتائج بالأسماء الكاملة ورسم المخططات """
        print("\n" + "═"*100)
        print(f"{'Algorithm Name':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Avg Time (ms)':<15}")
        print("─"*100)

        names = list(self.stats.keys())
        precision_vals, recall_vals, f1_vals, time_vals = [], [], [], []

        for name in names:
            s = self.stats[name]
            # الحسابات الإحصائية
            precision = s["True_Positives"] / (s["True_Positives"] + s["False_Positives"] + 1e-7)
            recall = s["True_Positives"] / (s["True_Positives"] + s["False_Negatives"] + 1e-7)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
            avg_time = np.mean(s["time"])

            precision_vals.append(precision)
            recall_vals.append(recall)
            f1_vals.append(f1)
            time_vals.append(avg_time)

            print(f"{name:<20} | {precision:<10.2f} | {recall:<10.2f} | {f1:<10.2f} | {avg_time:<15.2f}")

        self._plot_results(names, precision_vals, recall_vals, f1_vals, time_vals)

    def _plot_results(self, names, prec, rec, f1, times):
        plt.figure(figsize=(15, 6))
        
        # الرسمة الأولى: وقت المعالجة
        plt.subplot(1, 2, 1)
        bars = plt.bar(names, times, color=['#34495e', '#e67e22', '#27ae60'])
        plt.title('Average Processing Time per Image', fontweight='bold')
        plt.ylabel('Time in Milliseconds (ms)')
        plt.xticks(rotation=15)
        
        # الرسمة الثانية: مقاييس الدقة الكاملة
        plt.subplot(1, 2, 2)
        x = np.arange(len(names))
        width = 0.2
        plt.bar(x - width, prec, width, label='Precision', color='#3498db')
        plt.bar(x, rec, width, label='Recall', color='#9b59b6')
        plt.bar(x + width, f1, width, label='F1-Score', color='#2ecc71')
        
        plt.title('Accuracy Evaluation Metrics', fontweight='bold')
        plt.xticks(x, names, rotation=15)
        plt.ylim(0, 1.2)
        plt.legend()

        plt.tight_layout()
        plt.show()

# --- بدء التنفيذ ---
benchmarker = FaceDetectionBenchmarker(dataset_path="dataset_images")
benchmarker.run_full_benchmark() # سيعالج الـ 914 صورة بالكامل
benchmarker.generate_detailed_report()