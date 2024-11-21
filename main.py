import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tkinter import *
import customtkinter as ctk
import camera_detection as cd
import add_motion as am
from PIL import Image

# Create the main application window
app = ctk.CTk()
app.title("Translator Bahasa Isyarat BISINDO berbasis Machine Learning")
app.geometry("800x600")

# Create a button to open an image
logo_image = ctk.CTkImage(Image.open('logo.png'), size=(200,200)) # WidthxHeight

logo = ctk.CTkLabel(app, text="", image=logo_image)
logo.pack(pady=(80, 20))

judul = ctk.CTkLabel(app, text="Translator BISINDO berbasis Machine Learning", font=("Aptos", 20, "bold"))
judul.pack(pady=(10, 10))
deskripsi = ctk.CTkLabel(app, text="Program untuk mentranslate bahasa isyarat BISINDO menggunakan machine learning dengan TensorFlow LSTM dan input dari kamera OpenCV berfokus pada pengenalan gerakan tangan. Sistem ini memanfaatkan model LSTM untuk menganalisis urutan gerakan dan menerjemahkannya menjadi teks yang dapat dipahami.", font=("Aptos", 14), wraplength=700)
deskripsi.pack()

opencam_button = ctk.CTkButton(app, text="Mulai", command=lambda: cd.open_camera_and_compare(sequence_id=1), font=("Aptos", 14, "bold"))
opencam_button.pack(pady=(50,5))
addmotion_button = ctk.CTkButton(app, text="Tambah Gerakan", command=am.add_motion, fg_color="transparent", font=("Aptos", 12, "bold"), border_width=2, border_color="#3B8ED0")
addmotion_button.pack(pady=10)

app.mainloop()