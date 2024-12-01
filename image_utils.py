import os
import cv2
import gc

from PIL import Image
import numpy as np

face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

def crop_and_resize_face(image, path, confidence_threshold = 0.4):

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    h, w = image_cv.shape[:2]
    blob = cv2.dnn.blobFromImage(image_cv, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    if detections.shape[2] == 0:
        raise ValueError(f"No face detected in {path}")

    max_confidence = 0
    best_box = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence:
            max_confidence = confidence
            best_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

    if best_box is not None and max_confidence > confidence_threshold:
        x, y, x2, y2 = best_box.astype("int")
        face = image_cv[y:y2, x:x2]

        face_resized = cv2.resize(face, (256, 256))

        return Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

    raise ValueError(f"No face detected with sufficient confidence in {path}, best confidence: {max_confidence}")

def del_face_net():
    del face_net
    gc.collect()

if __name__ == '__main__':
    test_images = os.listdir('test_images')
    os.makedirs('cropped_images', exist_ok=True)
    
    for image in test_images:
        try:
            image_path = os.path.join('test_images', image)
            target_path = os.path.join('cropped_images', 'cropped_' + image)
            
            img = Image.open(image_path)
            cropped_img = crop_and_resize_face(img, image_path)
            cropped_img.save(target_path, 'JPEG')
        except ValueError as e:
            print(e)

