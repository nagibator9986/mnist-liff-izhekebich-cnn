import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageOps
theta_A = np.load("weights/theta_A.npy")
XeAe = np.load("weights/XeAe.npy")
print(f"theta_A.shape: {theta_A.shape}, XeAe.shape: {XeAe.shape}")

CANVAS_SIZE = 280 
IMG_SIZE = 20  


root = tk.Tk()
root.title("Распознавание цифр")

canvas = Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
canvas.grid(row=0, column=0, columnspan=2)

def clear_canvas():
    """ Очищает холст """
    canvas.delete("all")
    result_label.config(text="Распознанная цифра: ")

def softmax(x):
    """ Применяет softmax к массиву """
    exp_x = np.exp(x - np.max(x)) 
    return exp_x / np.sum(exp_x)

def snn_predict(img):
    """ Использует SNN для предсказания цифры """
    img_flat = img.flatten()
    print(f"img_flat.shape: {img_flat.shape}")
    
    if theta_A.shape[0] != img_flat.shape[0]:
        raise ValueError(f"Несовместимые размерности: theta_A.shape {theta_A.shape} и img_flat.shape {img_flat.shape}")
    
    XeAe_reduced = np.mean(XeAe, axis=1) if XeAe.ndim > 1 else XeAe 
    if XeAe_reduced.shape[0] != theta_A.shape[0]:
        raise ValueError(f"Несовместимые размерности: XeAe_reduced.shape {XeAe_reduced.shape} и theta_A.shape {theta_A.shape}")
    
    spikes = np.dot(theta_A, img_flat) + XeAe_reduced[:10]  
    print(f"spikes.shape: {spikes.shape}, spikes[:10]: {spikes[:10]}") 
    
    probs = softmax(spikes[:10])  
    prediction = np.argmax(probs) 
    return prediction

def predict_digit():
    """ Распознает нарисованную цифру """
    canvas.postscript(file="digit.ps", colormode='gray')  
    img = Image.open("digit.ps")
    img = img.convert('L')  
    img = ImageOps.invert(img)  
    img = img.resize((IMG_SIZE, IMG_SIZE)) 
    img = np.array(img) / 255.0 
    img = img.reshape(IMG_SIZE, IMG_SIZE)
    
    digit = snn_predict(img)
    
    result_label.config(text=f"Распознанная цифра: {digit}")
    root.update_idletasks()

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
