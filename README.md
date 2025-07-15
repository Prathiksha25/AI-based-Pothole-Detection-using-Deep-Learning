# AI-based-Pothole-Detection-using-Deep-Learning
Potholes on roads pose a significant risk to vehicles and public safety, contributing to traffic 
accidents, vehicle damage, and economic loss. Traditional pothole detection methods rely on 
manual inspections, which are often inefficient, time-consuming, and prone to human error. 
With advancements in artificial intelligence and computer vision, there is a growing 
opportunity to automate road defect detection using real-time video analysis and deep learning 
models. 
 
This project presents a lightweight and efficient pothole detection system built on the YOLOv8 
deep learning model. The model was trained on a custom-annotated dataset using Roboflow, 
with data augmentation techniques applied to improve robustness under varying lighting and 
environmental conditions. The system is capable of performing real-time inference on video 
feeds from webcams or dashcams and overlays detection results with bounding boxes and 
confidence scores. All detected events are logged both locally via CSV and remotely to 
Firebase Realtime Database, enabling centralized data monitoring and analysis. 
 
The application includes a user-friendly graphical interface built with Tkinter and OpenCV, 
allowing users to start detection, view results live, and access logged data. This system aims to assist municipalities and transport authorities in 
automating road maintenance and improving roadway safety.
