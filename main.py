import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from ultralytics import YOLO
from tkinter import Tk, filedialog
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PyQt5.QtWidgets import QApplication, QFileDialog

model_elbow_frac = tf.keras.models.load_model("D:\Prajith K\Studies\Projects\Final Project\Project\Multi_Model_Medical_Image_Classification_Detection\Weights\ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("D:\Prajith K\Studies\Projects\Final Project\Project\Multi_Model_Medical_Image_Classification_Detection\Weights\ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("D:\Prajith K\Studies\Projects\Final Project\Project\Multi_Model_Medical_Image_Classification_Detection\Weights\ResNet50_Shoulder_frac.h5")

categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

def predict(img, model="Parts"):
    size = 224
    if model == 'Parts':
        model = YOLO("D:\Prajith K\Studies\Projects\Final Project\Project\Multi_Model_Medical_Image_Classification_Detection\Weights\YOLO_Body_Parts.pt")
        names = {0: 'Brain', 1: 'Elbow', 2: 'Hand', 3: 'Shoulder'}
        result = model.predict(img, verbose=False)
        ans = result[0].probs.top1
        return names[ans]        
    else:
        if model == 'Brain':
            model = YOLO("D:\Prajith K\Studies\Projects\Final Project\Project\Multi_Model_Medical_Image_Classification_Detection\Weights\YOLO_Brain_detect.pt")
            names = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
            result = model.predict(img, verbose=False)
            ans = result[0].probs.top1
            return names[ans]
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac

        temp_img = image.load_img(img, target_size=(size, size))
        x = image.img_to_array(temp_img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        prediction = chosen_model.predict(images)
        prediction = np.argmax(prediction, axis=1)
        prediction_str = categories_fracture[prediction.item()]
        return prediction_str

def get_image_paths():
    app = QApplication([])
    file_dialog = QFileDialog()
    file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
    file_dialog.setViewMode(QFileDialog.Detail)
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    if file_dialog.exec_():
        file_paths = file_dialog.selectedFiles()
        return file_paths
    else:
        return None

def generate_pdf(img_paths, results, save_path):
    c = canvas.Canvas(save_path, pagesize=letter)
    for img_path, result in zip(img_paths, results):
        img_name = os.path.basename(img_path)
        img = ImageReader(img_path)
        ans = predict(img_path)
        
        if ans == 'Elbow':
            img_type = "Elbow.Xray"
        elif ans == 'Hand':
            img_type = "Hand.Xray"
        elif ans == 'Shoulder':
            img_type = "Shoulder.Xray"
        elif ans == 'Brain':
            img_type = "Brain.MRI"
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, 740,"                            ")
        c.drawString(50, 730,"                            ")
        c.drawString(50, 700, f"  Image Name: {img_name}")
        c.drawString(50, 660, f"  Given Medical Image:- {img_type}")
        
        if result == 'fractured':
            c.drawString(50, 620, "  Condition:- Fractured") 
        elif result == 'normal':
            c.drawString(50, 620, "  Condition:- Normal")
        elif result == 'glioma':
            c.drawString(50, 620, "  Condition:- Tumor Detected")
            c.drawString(50, 580, "  Type of Tumor:- Glioma")
        elif result == 'meningioma':
            c.drawString(50, 620, "  Condition:- Tumor Detected")
            c.drawString(50, 580, "  Type of Tumor:- Meningioma")
        elif result == 'notumor':
            c.drawString(50, 620, "  Condition:- No Tumor Detected")
        elif result == 'pituitary':
            c.drawString(50, 620, "  Condition:- Tumor Detected")
            c.drawString(50, 580, "  Type of Tumor:- Pituitary")

        c.drawImage(img, 50, 190, width=400, height=325, preserveAspectRatio=True)
        c.showPage()
    c.save()

def main():
    img_paths = get_image_paths()
    if img_paths:
        results = []
        for img_path in img_paths:
            ans = predict(img_path)
            res = predict(img_path, ans)
            results.append(res)

        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        generate_pdf(img_paths, results, save_path)
        print(" ")
        print(" ")
        print("PDF generated successfully.")
    else:
        print("No images selected. Exiting...")

if __name__ == "__main__":
    main()
