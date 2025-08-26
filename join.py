import cv2 as cv
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import sympy as sp
import easyocr
import  matplotlib.pyplot as plt


class digits:
    def __init__(self):
        self.model = load_model("digit_operator_model.h5")
        self.reader = easyocr.Reader(['en'])
        self.digit_classes = list(range(10))  # 0-9 only

    def preprocess_digit(self, img):
        img = cv.resize(img, (28, 28))
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 28, 28, 1)
        return img

    def is_digit_confident(self, crop):
        pred = self.model.predict(self.preprocess_digit(crop), verbose=0)[0]
        class_id = np.argmax(pred)
        confidence = pred[class_id]
        return class_id, confidence


    def segment_characters(self, image):
        # If it's a file path, read image
        if isinstance(image, str):
            img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        else:
            img = image

        # If still colored, convert to grayscale
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Threshold to binary
        _, thresh = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
        thresh = thresh.astype(np.uint8)

        # Find contours for each character
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        char_images = []

        for cnt in sorted(contours, key=lambda c: cv.boundingRect(c)[0]):
            x, y, w, h = cv.boundingRect(cnt)
            if w > 5 and h > 5:  # Ignore small noise
                char_crop = img[y:y+h, x:x+w]
                char_images.append((char_crop, (x, y, w, h)))

        return char_images



    def recognize_expression(self, img):
        char_images = self.segment_characters(img)  # returns list of (char_img, bbox)
        result_str = ""
        result_labels = []
        result_imgs = []
        result_boxes = []

        for char_img, bbox in char_images:
            class_id, conf = self.is_digit_confident(char_img)

            if class_id in self.digit_classes and conf > 0.85:  # Digit
                result_str += str(class_id)
                result_labels.append(str(class_id))
            else:  # Operator
                op_text = self.reader.readtext(char_img, detail=0)
                if op_text:
                    result_str += op_text[0]
                    result_labels.append(op_text[0])
                else:
                    result_str += "?"
                    result_labels.append("?")
            
            # Save normalized image for plotting
            norm_img = cv.resize(char_img, (28, 28))
            if norm_img.max() > 1:
                norm_img = norm_img / 255.0
            result_imgs.append(norm_img)

            # Save bounding box
            result_boxes.append(bbox)
            if result_str.endswith("_"):
                result_str = result_str[:-1]

        return result_str, result_labels, result_imgs, result_boxes
    


