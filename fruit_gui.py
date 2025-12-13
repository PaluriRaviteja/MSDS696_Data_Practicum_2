# fruit_gui_FINAL.py
# Complete working GUI with drawing + upload

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.simpledialog as sd
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import os
import random
import glob

MODEL_PATH = "fruit_cnn.h5"
IMG_SIZE = 64
LINE_WIDTH = 6
CLASS_NAMES = ['apple', 'banana', 'grapes', 'pineapple', 'strawberry']

class FruitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Sketch Recognizer")
        
        # Load model
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("✓ Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model:\n{e}")
            root.quit()
            return
        
        # Drawing state
        self.canvas_size = 400
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.uploaded_mode = False  # Track if we're showing an uploaded image
        
        # PIL image for drawing
        self.pil_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
        # UI Setup
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        self.pred_label = tk.Label(
            self.root, 
            text="Draw a fruit or upload an image",
            font=("Arial", 16, "bold"),
            fg="darkblue"
        )
        self.pred_label.pack(pady=10)
        
        # Canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            cursor="cross",
            relief="solid",
            borderwidth=2
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Bind drawing events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Upload Image", command=self.upload_image).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Predict", command=self.predict).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_canvas).grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="Show Sample", command=self.show_sample).grid(row=0, column=3, padx=5)
        ttk.Button(btn_frame, text="Quit", command=self.root.quit).grid(row=0, column=4, padx=5)
        
        # Tip
        tip = tk.Label(
            self.root,
            text="Tip: Draw simple & fast (QuickDraw style) or upload a QuickDraw image!",
            font=("Arial", 10),
            fg="gray"
        )
        tip.pack(pady=5)
    
    def on_press(self, event):
        self.drawing = True
        self.uploaded_mode = False  # Switch back to drawing mode
        self.last_x = event.x
        self.last_y = event.y
    
    def on_drag(self, event):
        if not self.drawing:
            return
        
        x, y = event.x, event.y
        
        # Draw on canvas
        self.canvas.create_line(
            self.last_x, self.last_y, x, y,
            fill="black", width=LINE_WIDTH,
            capstyle=tk.ROUND, smooth=True
        )
        
        # Draw on PIL image
        self.pil_draw.line(
            [self.last_x, self.last_y, x, y],
            fill="black", width=LINE_WIDTH
        )
        
        self.last_x = x
        self.last_y = y
    
    def on_release(self, event):
        self.drawing = False
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.uploaded_mode = False
        self.pred_label.config(text="Draw a fruit or upload an image")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select a fruit image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Load image
            img = Image.open(file_path)
            
            # Clear and set uploaded mode
            self.clear_canvas()
            self.uploaded_mode = True
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to fit canvas
            img.thumbnail((self.canvas_size, self.canvas_size), Image.LANCZOS)
            
            # Center on canvas
            w, h = img.size
            x_offset = (self.canvas_size - w) // 2
            y_offset = (self.canvas_size - h) // 2
            
            # Update PIL image
            self.pil_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
            self.pil_image.paste(img, (x_offset, y_offset))
            self.pil_draw = ImageDraw.Draw(self.pil_image)
            
            # Display on canvas
            photo = ImageTk.PhotoImage(self.pil_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            
            self.pred_label.config(text="Image uploaded! Click Predict.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{e}")
    
    def show_sample(self):
        """Show a random training sample"""
        fruit = random.choice(CLASS_NAMES)
        pattern = os.path.join("data_quickdraw", fruit, "*.png")
        files = glob.glob(pattern)
        
        if not files:
            # Try data_fruits as fallback
            pattern = os.path.join("data_fruits", fruit, "*.png")
            files = glob.glob(pattern)
        
        if not files:
            messagebox.showinfo("No samples", f"No images found for {fruit}")
            return
        
        sample_path = random.choice(files)
        
        # Load and display
        try:
            img = Image.open(sample_path)
            
            # Show in popup
            popup = tk.Toplevel(self.root)
            popup.title(f"Sample: {fruit}")
            
            display = img.resize((300, 300), Image.NEAREST)
            photo = ImageTk.PhotoImage(display)
            
            label = tk.Label(popup, image=photo)
            label.image = photo
            label.pack(padx=10, pady=10)
            
            tk.Label(
                popup, 
                text=f"This is a '{fruit}' from training data\nDraw similar simple sketches!",
                font=("Arial", 11)
            ).pack(pady=5)
            
            ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load sample:\n{e}")
    
    def preprocess(self, img):
        """Preprocess image for model"""
        # Convert to grayscale
        img = img.convert("L")
        
        # Resize to 64x64
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        
        # Convert to array
        arr = np.array(img).astype("uint8")
        
        # Threshold (same as training)
        _, binary = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY)
        
        # Normalize
        binary = binary.astype("float32") / 255.0
        
        # Reshape
        binary = binary.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        return binary
    
    def predict(self):
        """Predict the current image"""
        # Get current image
        img = self.pil_image.copy()
        
        # Check if empty (only if not uploaded)
        if not self.uploaded_mode:
            img_gray = img.convert("L")
            img_array = np.array(img_gray)
            if np.mean(img_array) > 250:
                self.pred_label.config(text="⚠️ Canvas is empty! Draw something first.")
                return
        
        # Preprocess
        x = self.preprocess(img)
        
        # Save debug image
        debug_img = (x[0, :, :, 0] * 255).astype(np.uint8)
        Image.fromarray(debug_img).save("debug_last_prediction.png")
        print("Saved debug_last_prediction.png")
        
        # Predict
        preds = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        prob = float(preds[idx])
        label = CLASS_NAMES[idx]
        
        # Show result
        self.pred_label.config(text=f"Prediction: {label.upper()} ({prob*100:.1f}%)")
        
        # Print top 3
        top3 = np.argsort(preds)[-3:][::-1]
        top3_str = ", ".join([f"{CLASS_NAMES[i]}: {preds[i]*100:.0f}%" for i in top3])
        print(f"Top 3: {top3_str}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FruitGUI(root)
    root.mainloop()
