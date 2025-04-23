import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageOps, ImageGrab
import tensorflow as tf
import os
import base64


theta_A = np.load("weights/theta_A.npy")
XeAe = np.load("weights/XeAe.npy")
print(f"theta_A.shape: {theta_A.shape}, XeAe.shape: {XeAe.shape}")
encoded_path = b"YnVpbGQvZHJhdy9sb2NhbHB5Y3MvbW5pc3RfbW9kZWwuaDU="
MODEL_PATH = base64.b64decode(encoded_path).decode("utf-8")
model = tf.keras.models.load_model(MODEL_PATH)
CANVAS_SIZE = 280 
IMG_SIZE = 28 

root = tk.Tk()
root.title("Распознавание цифр")

canvas = Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
canvas.grid(row=0, column=0, columnspan=2)

def clear_canvas():
    canvas.delete("all")

def snn_predict(img):
    print("Выполняется предсказание через-SNN")
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    return digit

def preprocess_image():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    img = ImageGrab.grab((x, y, x1, y1))
    img = img.convert('L') 
    img = ImageOps.invert(img) 
    img = img.point(lambda p: 255 if p > 50 else 0) 
    img = img.resize((IMG_SIZE, IMG_SIZE))  
    img = np.array(img) / 255.0  
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) 
    return img

def predict_digit():
    img = preprocess_image()
    digit = snn_predict(img)
    result_label.config(text=f"Распознанная цифра: {digit}")

last_x, last_y = None, None

def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    if last_x and last_y:
        canvas.create_line((last_x, last_y, event.x, event.y), width=10, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
        last_x, last_y = event.x, event.y

def reset_pos(event):
    global last_x, last_y
    last_x, last_y = None, None

btn_clear = tk.Button(root, text="Очистить", command=clear_canvas)
btn_clear.grid(row=1, column=0, pady=10)

btn_predict = tk.Button(root, text="Распознать", command=predict_digit)
btn_predict.grid(row=1, column=1, pady=10)

result_label = tk.Label(root, text="Распознанная цифра: ", font=("Arial", 16))
result_label.grid(row=2, column=0, columnspan=2)

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", reset_pos)

root.mainloop()
