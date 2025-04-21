from flask import Flask, render_template, request, redirect
import os
import uuid  # Unique filenames
import cv2
import numpy as np
import easyocr
import util

# Initialize Flask app
app = Flask(__name__)

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure directories dynamically
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define paths for YOLO model files dynamically
MODEL_CFG_PATH = os.path.join(BASE_DIR, "model", "cfg", "darknet-yolov3.cfg")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "weights", "model.weights")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class.names")

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [j.strip() for j in f.readlines() if len(j.strip()) > 0]

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(MODEL_CFG_PATH, MODEL_WEIGHTS_PATH)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def process_image(image_path, filename):
    """Processes an image to detect a number plate and extract text using EasyOCR."""
    image = cv2.imread(image_path)

    if image is None:
        return None, None, "Error: Unable to read image."


    H, W, _ = image.shape

    # Convert image for YOLO model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = util.get_outputs(net)

    # Process detections
    bboxes = []
    scores = []
    class_ids = []

    for detection in detections:
        bbox = detection[:4]
        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]
        bbox_confidence = detection[4]
        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # Apply Non-Maximum Suppression (NMS)
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    number_plate_cnt = None
    cropped_filename = None
    extracted_text = "Number plate not detected."

    for bbox in bboxes:
        xc, yc, w, h = bbox
        x1, y1, x2, y2 = int(xc - w / 2), int(yc - h / 2), int(xc + w / 2), int(yc + h / 2)

        # Crop the license plate region
        license_plate = image[y1:y2, x1:x2].copy()
        if license_plate.size == 0:
            continue

        # Save the detected plate with a unique name
        cropped_filename = f"plate_{uuid.uuid4().hex}.png"
        cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
        cv2.imwrite(cropped_path, license_plate)

        # Convert to grayscale and apply thresholding for better OCR
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        # Save the thresholded image
        threshold_filename = f"threshold_{filename}"
        threshold_path = os.path.join(app.config['UPLOAD_FOLDER'], threshold_filename)
        cv2.imwrite(threshold_path, license_plate_thresh)

        # Use EasyOCR to extract text
        output = reader.readtext(license_plate_thresh)
        extracted_text_lines = []
        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0.4:
                extracted_text_lines.append(text)

        extracted_text = " ".join(extracted_text_lines) if extracted_text_lines else "Number plate not detected."

        if extracted_text != "Number plate not detected.":
            number_plate_cnt = [[x1, y1], [x2, y2]]
            break  # Stop after finding the first valid plate

    # Save processed image with detected plate
    processed_filename = f"processed_{filename}"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

    if number_plate_cnt is not None:
        # Draw bounding box on a copy of the original image
        processed_image = image.copy()
        cv2.rectangle(processed_image, tuple(number_plate_cnt[0]), tuple(number_plate_cnt[1]), (0, 255, 0), 3)

        # Save with high quality
        cv2.imwrite(processed_path, processed_image)

    return processed_filename, cropped_filename, extracted_text, threshold_filename
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            processed_filename, cropped_filename, extracted_text, threshold_filename = process_image(file_path, unique_filename)

            return render_template("result.html", 
                                   result=extracted_text, 
                                   uploaded_image=unique_filename, 
                                   processed_image=processed_filename, 
                                   cropped_image=cropped_filename,
                                   threshold_image=threshold_filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=5000, debug=True)