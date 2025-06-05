import cv2
import face_recognition
import os
import numpy as np
from unidecode import unidecode

# Yüz fotoğraflarının bulunduğu klasör
path = 'photos'
known_face_encodings = []
known_face_names = []

# Klasördeki her yüzü öğren
for filename in os.listdir(path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(path, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:  # boş dönmemesi için kontrol
            known_face_encodings.append(encoding[0])
            # Dosya adından uzantıyı çıkar, Türkçe karakterleri ASCII'ye çevir
            name = os.path.splitext(filename)[0]
            name = unidecode(name)  # Örn: "Çağrı" → "Cagri"
            known_face_names.append(name)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Boyutu küçült (daha hızlı işlem için)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Yüzleri ve yüz encoding'lerini bul
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        name = "Bilinmiyor"

        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Orijinal boyuta göre koordinatları ölçekle
        top, right, bottom, left = [v * 4 for v in face_location]

        # Dikdörtgen çiz ve ASCII'ye çevrilmiş ismi yaz
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Yüz Tanıma", frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
