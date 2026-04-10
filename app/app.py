import os
import json
import uuid
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, url_for

BASE_DIR = r"C:\Users\visha\OneDrive\Desktop\Agro_Saarthi"
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
EN_JSON_PATH = os.path.join(BASE_DIR, "languages", "en.json")
HI_JSON_PATH = os.path.join(BASE_DIR, "languages", "hi.json")
with open(EN_JSON_PATH, "r", encoding="utf-8") as f:
    en_data = json.load(f)

with open(HI_JSON_PATH, "r", encoding="utf-8") as f:
    hi_data = json.load(f)

APP_DIR = os.path.join(BASE_DIR, "app")
STATIC_DIR = os.path.join(APP_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)

app = Flask(__name__, template_folder="templates", static_folder="static")

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

TREATMENT_MAP = {
    "Potato___Early_blight": "Remove infected leaves and apply a recommended fungicide. Maintain proper spacing and avoid overhead watering.",
    "Potato___Late_blight": "Remove infected parts immediately, avoid water stagnation, and use a suitable fungicide under expert guidance.",
    "Potato___healthy": "Leaf appears healthy. Continue regular crop monitoring and proper nutrient management.",
    "Tomato___Early_blight": "Prune affected leaves, improve air circulation, and use preventive fungicide spray if needed.",
    "Tomato___Late_blight": "Remove infected leaves/fruits, reduce excess moisture, and apply suitable disease control measures promptly.",
    "Tomato___healthy": "Leaf appears healthy. Maintain regular care, irrigation balance, and field hygiene."
}


def preprocess_image(image_path):
    orig = cv2.imread(image_path)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(orig_rgb, IMG_SIZE)
    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
    return orig, img_array


def generate_gradcam(image_path, pred_idx):
    orig, img_array = preprocess_image(image_path)

    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    last_conv_layer_obj = model.get_layer(last_conv_layer)
    last_conv_index = model.layers.index(last_conv_layer_obj)

    conv_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer_obj.output
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer_obj.output.shape[1:])
    x = classifier_input

    for layer in model.layers[last_conv_index + 1:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        loss = predictions[:, pred_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)

    output_name = f"{uuid.uuid4().hex}_gradcam.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(output_path, overlay)

    return output_name

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/language")
def language():
    return render_template("language.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    language = request.form.get("lang", "en")
    return render_template("login.html", lang=language)

@app.route("/home", methods=["GET", "POST"])
def index():
    language = request.form.get("lang", "en")
    name = request.form.get("name", "")

    if request.method == "POST" and "leaf_image" in request.files:
        print("Selected Language:", language)

        file = request.files["leaf_image"]

        if file.filename == "":
            return "No file selected."

        ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        file.save(save_path)

        _, img_array = preprocess_image(save_path)

        preds = model.predict(img_array, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = class_names[pred_idx]
        confidence = float(preds[pred_idx])

        gradcam_file = generate_gradcam(save_path, pred_idx)
        if language == "hi":
            display_prediction = hi_data["disease_names"].get(pred_class, pred_class)
            treatment_steps = hi_data["treatments"].get(pred_class, ["उपचार उपलब्ध नहीं है।"])
            prevention_steps = hi_data["prevention_tips"].get(pred_class, [])
        else:
            display_prediction = en_data["disease_names"].get(pred_class, pred_class)
            treatment_steps = en_data["treatments"].get(pred_class, ["No treatment available."])
            prevention_steps = en_data["prevention_tips"].get(pred_class, [])

       
        # --- New Severity Logic (Image-Based) ---
        if "healthy" in pred_class.lower():
            severity = "Healthy"
        else:
            image = cv2.imread(save_path)

            if image is None:
                severity = "Moderate"
            else:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                lower_brown = np.array([5, 40, 20])
                upper_brown = np.array([25, 255, 200])

                lower_yellow = np.array([20, 40, 40])
                upper_yellow = np.array([40, 255, 255])

                brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
                yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

                disease_mask = cv2.bitwise_or(brown_mask, yellow_mask)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, leaf_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

                diseased_pixels = cv2.countNonZero(disease_mask)
                total_leaf_pixels = cv2.countNonZero(leaf_mask)

                if total_leaf_pixels == 0:
                    severity = "Moderate"
                else:
                    ratio = diseased_pixels / total_leaf_pixels

                    if ratio < 0.15:
                        severity = "Mild"
                    elif ratio < 0.35:
                        severity = "Moderate"
                    else:
                        severity = "Severe"

        return render_template(
            "result.html",
            uploaded_image=url_for("static", filename=f"uploads/{unique_name}"),
            gradcam_image=url_for("static", filename=f"outputs/{gradcam_file}"),
            prediction=display_prediction,
            confidence=round(confidence * 100, 2),
            treatment_steps=treatment_steps,
            prevention_steps=prevention_steps,
            lang=language,
            severity=severity 

        )

    return render_template("index.html", lang=language, name=name)
if __name__ == "__main__":
    app.run(debug=True)