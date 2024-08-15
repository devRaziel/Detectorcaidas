import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
import telebot
from geopy.geocoders import Nominatim
from flask import Flask, render_template, Response, request, jsonify
import sqlite3
import hashlib
from flask_cors import CORS
import os
import time
import serial

# Flask app setup
app = Flask(__name__)

camera = cv2.VideoCapture("ci.mp4")

# YOLO model and SORT tracker setup
model = YOLO(r"yolov8n-pose.pt")
tracker = Sort()

# Telegram Bot setup
TOKEN = '5955691907:AAHr3oFVhPDRM8LZgkln2x2L0r5uXdiUY0c'
bot = telebot.TeleBot(TOKEN)

# Global variables for alert timing
ultima_ejecucion = 0

def obtener_direccion(latitud, longitud):
    geolocalizador = Nominatim(user_agent="mi_aplicacion")
    ubicacion = geolocalizador.reverse((latitud, longitud))
    if ubicacion:
        print(f"Dirección encontrada: {ubicacion.address}")
        return ubicacion.address
    else:
        print("Dirección no encontrada.")
        return "Dirección no encontrada."

# Geolocation setup
latitud = -0.905863
longitud = -78.621728
direccion = obtener_direccion(latitud, longitud)

def enviar_alerta(chat_id, mensaje):
    global ultima_ejecucion
    tiempo_actual = time.time()

    if tiempo_actual - ultima_ejecucion >= 10:
        bot.send_message(chat_id, mensaje)
        ultima_ejecucion = tiempo_actual
        with open("images/persona_detectada_1.jpg", 'rb') as photo:
            bot.send_photo(chat_id, photo)
    else:
        print("La función enviar_alerta no puede ejecutarse más de una vez en 10 segundos.")

def gen_frames():
    prev_y0 = None
    prev_time = time.time()
    john = 0
    cap = cv2.VideoCapture("ci.mp4")
    
    y_line = 0
    y_line2 = 0
    # Abre el archivo en modo 'append' ('a') para agregar contenido sin sobrescribir
   

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        results = model(frame, show=False, show_boxes=False, show_labels=False, stream=True)

        for res in results:
            annotated_frame = res.plot(labels=False, conf=False, boxes=False)
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0]
            with open('archivofffff.txt', 'w') as archivo:
                if len(res.boxes.cls) > 0:
        # Acceder al primer elemento del tensor
                    
                    print("Primer elemento del tensor:")
                    with open('archivofffff.txt', 'w') as archivo:
        # Escribir en el archivo
                        archivo.write("0")
                else:
                    print("El tensor está vacío y no se puede acceder a su contenido.")
                    archivo.write("1")
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)
            kpts = res.keypoints

            y_line = frame.shape[0] * 4 // 8
            y_line2 = frame.shape[0] * 7 // 8

            for track_id, (xmin, ymin, xmax, ymax) in zip(tracks[:, 4], tracks[:, :4]):
                w = xmax - xmin
                h = ymax - ymin
                carpeta_regiones = "images"
                ruta_archivo = os.path.join(carpeta_regiones, f"persona_detectada_{track_id}.jpg")
                region_persona = frame[ymin:ymax, xmin:xmax]

                if prev_y0 is not None:
                    current_y0 = kpts.xy[0, 0][1]
                    delta_y0 = abs(current_y0 - prev_y0)
                    current_time = time.time()
                    delta_time = current_time - prev_time
                    john = delta_y0 / delta_time
                    if delta_time > 0 and delta_y0 / delta_time > 100:
                        print("Cambio brusco detectado en el punto 0.")

                if any(kpts.xy[0, idx][1] > y_line for idx in [0, 1, 2, 3, 4, 5, 6, 11, 12]) and john > 120 and w / h > 0.7:
                    cv2.putText(annotated_frame, f"Inestable (falling): {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
                elif w / h > 0.7 and any(kpts.xy[0, idx][1] > y_line2 for idx in [0, 1, 2, 3, 4, 5, 6, 11, 12]):
                    cv2.putText(annotated_frame, f"Lying Down (fall): {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                    cv2.imwrite(ruta_archivo, region_persona)
                    enviar_alerta('1338645187', "Caida detectada adulto mayor "+"Localizacion: "+str(direccion))
                    
                elif any(kpts.xy[0, idx][1] < y_line2 for idx in [0, 1, 2, 3, 4, 5, 6, 11, 12]) and w / h > 0.7:
                    cv2.putText(annotated_frame, f"Acostado: {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                else:
                    cv2.putText(annotated_frame, f"Estable: {track_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                prev_y0 = kpts.xy[0, 0][1]
                prev_time = time.time()


            cv2.line(annotated_frame, (0, y_line), (frame.shape[1], y_line), (255, 0, 0), 2)
            cv2.line(annotated_frame, (0, y_line2), (frame.shape[1], y_line2), (255, 0, 0), 2)

        annotated_frame = cv2.resize(annotated_frame, (800, 600))
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video2', methods=['GET', 'POST'])
def index2():
    return render_template('index2.html')
@app.route('/get_info', methods=['GET'])
def get_info():
    conn = sqlite3.connect('imagenes.db')  # Cambia el nombre del archivo de base de datos si es diferente
    c = conn.cursor()

    c.execute('SELECT id, cont_malas, cont_medias, cont_golden FROM imagenes')  # Selecciona solo las columnas necesarias
    cdr_data = c.fetchall()
    
    conn.close()

    return jsonify(cdr_data)
if __name__ == '__main__':
    app.run(debug=True)
