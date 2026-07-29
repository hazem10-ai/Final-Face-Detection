# Final Face Detection

A Python-based face detection project that compares three widely used face detection algorithms:

- Haar Cascade
- HOG + Linear SVM
- Deep Learning (SSD + ResNet DNN)

The project evaluates each algorithm using a dataset of 914 images under different lighting conditions, face poses, and multiple-face scenarios.

---

## Features

- Face detection using three different algorithms.
- Performance comparison based on:
  - Precision
  - Recall
  - F1-Score
  - Processing Time
- Easy-to-use Python implementation.
- Suitable for educational and research purposes.

---

## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib

---

## Dataset

- Total Images: **914**
- Various lighting conditions
- Different face poses
- Multiple face scenarios

> The complete dataset is not included in this repository due to GitHub storage limitations.

---

## Results

| Algorithm | Precision | Recall | F1-Score | Time (ms) |
|-----------|----------:|-------:|----------:|----------:|
| Haar Cascade | 0.93 | 1.00 | 0.96 | 38.15 |
| HOG + SVM | **0.96** | 0.99 | **0.98** | **31.20** |
| Deep Learning (DNN) | 0.90 | 1.00 | 0.95 | 45.85 |

HOG + SVM achieved the best overall performance, providing the highest F1-score and the fastest processing time.

---

## Project Structure

```
Final-Face-Detection
│
├── Project.py
├── models/
├── screenshots/
├── docs/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python Project.py
```

---

## Author

Hazem Mansour
