# Speech Command Recognition with Deep Learning  
**تشخیص دستورات صوتی با یادگیری عمیق**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=Python&logoColor=white)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)  
[![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=Keras&logoColor=white)](https://keras.io/)  
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243?logo=numpy&logoColor=white)](https://numpy.org/)  
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-11557C?logo=image%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAAA+0lEQVRIie2UMY6CQBBFnwURrSwsbGwUjY2JxsbCQk/grj2BlYWFjYVH8ARWNp4ACwvBGAsLvYCNJ7AxkRQYFhh2lWiykZ+0ZN7MvDeTX4H/ThQCPgkh9jHGtwvhui5jjH0pZRuArlJqB0DHcXZaa2ZZ1svz/ACgDcCyLGZZ1svz/AAAlFJb13UPAE7TND+O4zDDMF5lWVZ1Xc+01mwYhhetdbter/dSSg4AIcRKa30GgMbj8X0+n78AoLW+zmazFwBwzvm1KIp7mqYnrXU7iqJ7nucnrTWjlHIAQJqmHJHnedfpdLr5vo8wDN+73e4tjxACABBC4DzPAQCmaX4BwPM8HkXRVUq5B4CiKD4ppRwAlFJ/AKBer1cAQJqm7yAILgDAOX8xxhgAlGVZ1XX9AwD/AKMCF1yr8wAAAABJRU5ErkJggg==)](https://matplotlib.org/)  
[![PyAudio](https://img.shields.io/badge/PyAudio-0.2.12-0077B5?logo=Python&logoColor=white)](https://pypi.org/project/PyAudio/)

---

## Table of Contents (English)
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)

---

## فهرست مطالب (فارسی)
1. [مقدمه](#مقدمه)
2. [ویژگی‌های اصلی](#ویژگی‌های-اصلی)
3. [تکنولوژی‌های استفاده‌شده](#تکنولوژی‌های-استفاده‌شده)
4. [نصب و راه‌اندازی](#نصب-و-راه‌اندازی)

---

## Introduction
This project implements a deep learning model for speech command recognition using convolutional neural networks (CNN). The system processes audio input, converts it to spectrograms, and classifies it into predefined commands. Features include:

- Audio preprocessing with STFT spectrogram conversion
- Configurable hyperparameters via JSON configuration
- Real-time audio recording and prediction
- Model evaluation with accuracy metrics and confusion matrix
- Cross-platform compatibility

---

## مقدمه
این پروژه یک مدل یادگیری عمیق برای تشخیص دستورات صوتی با استفاده از شبکه‌های عصبی کانولوشنی (CNN) پیاده‌سازی می‌کند. سیستم ورودی صوتی را پردازش کرده، آن را به طیف‌نگار تبدیل می‌کند و به دسته‌های از پیش تعریف شده طبقه‌بندی می‌کند. ویژگی‌ها شامل:

- پیش‌پردازش صوتی با تبدیل به طیف‌نگار STFT
- پیکربندی پارامترها از طریق فایل JSON
- ضبط صدا و پیش‌بینی بلادرنگ
- ارزیابی مدل با معیارهای دقت و ماتریس درهم‌ریختگی
- سازگاری با پلتفرم‌های مختلف

---

## Key Features / ویژگی‌های اصلی
- **Spectrogram Conversion / تبدیل به طیف‌نگار:**  
  Converts raw audio to time-frequency representations
  
- **CNN Model / مدل شبکه عصبی کانولوشنی:**  
  5-layer architecture with dropout and normalization
  
- **Configurable Parameters / تنظیم پارامترها:**  
  Batch size, epochs, validation split via config.json
  
- **Real-time Recording / ضبط بلادرنگ:**  
  Built-in audio recording capability
  
- **Evaluation Metrics / معیارهای ارزیابی:**  
  Accuracy, loss curves, and confusion matrix visualization

---

## Technologies Used / تکنولوژی‌های استفاده‌شده
- **TensorFlow 2.x**
- **Keras 3.x**
- **NumPy**
- **Matplotlib**
- **PyAudio**
- **Seaborn**
- **IPython**

---

## Installation / نصب و راه‌اندازی

### English
1. **Clone Repository:**
```bash
git clone https://github.com/yourusername/speech-recognition.git
cd speech-recognition
