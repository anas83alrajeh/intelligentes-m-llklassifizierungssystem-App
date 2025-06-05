# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 13:03:49 2025

@author: anasa
"""

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class_names_de = ['Karton', 'Glas', 'Metall', 'Papier', 'Kunststoff', 'Müll']

# Modell laden
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('models/waste_classifier.pth', map_location=device))
model = model.to(device)
model.eval()

# Streamlit-UI
st.title("♻️ Abfallklassifikation mit KI")
st.write("Lade ein Bild hoch, und das Modell erkennt die Abfallart.")

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='📷 Hochgeladenes Bild', use_column_width=True)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1)
        st.success(f"🗑️ Erwartete Klasse: **{class_names[pred]}**")
        st.write(f"Deutsch: **{class_names_de[pred]}**")

# Info am Ende der Seite
st.markdown("---")
st.markdown("Erstellt von Anas Al Rajeh  \nKontakt: anasalrajeh9@gmail.com")
