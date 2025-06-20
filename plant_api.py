from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
import tempfile
import os
import numpy as np
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_plant_disease(crop_name, image_path):
    """
    Predicts the disease for the given crop and image.
    crop_name: one of 'cotton', 'maize', 'rice', 'tomato', 'wheat'
    image_path: path to the image file
    Returns: (result, confidence, extra_info)
    """
    crop_name = crop_name.lower()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if crop_name == 'cotton':
        model = load_model("cotton_plant_disease_classifier.h5", compile=False)
        img = image.load_img(image_path, target_size=(180, 180))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]
        exp_pred = np.exp(prediction - np.max(prediction))
        probs = exp_pred / np.sum(exp_pred)
        class_label = np.argmax(probs)
        class_names = ['Aphids', 'Army worm', 'Bacterial Blight', 'Healthy leaf', 'Powdery Mildew', 'Target spot']
        result = class_names[class_label]
        confidence = float(np.max(probs))
        return result, confidence, None

    elif crop_name == 'maize':
        class_labels = [
            'Cercospora leaf spot (Gray leaf spot)',
            'Common rust',
            'Northern Leaf Blight',
            'healthy'
        ]
        model = load_model('maize_disease_detection_new_model.h5', compile=False)
        img = image.load_img(image_path, target_size=(244, 244))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype('float32') / 255.0
        prediction = model.predict(x)
        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        disease_name = class_labels[predicted_index]
        return disease_name, confidence, None

    elif crop_name == 'rice':
        model = load_model("riceplantdetectionmodel.h5", compile=False)
        img = image.load_img(image_path, target_size=(300, 300))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]
        class_label = np.argmax(prediction)
        class_names = ['Bacterialblight', 'Brownspot', 'Leafsmut']
        result = class_names[class_label]
        confidence = float(np.max(prediction))
        return result, confidence, None

    elif crop_name == 'tomato':
        class_labels = [
            "Bacterial_spot",
            "Early_blight",
            "Late_blight",
            "Leaf_Mold",
            "Septoria_leaf_spot",
            "Spider_mites Two-spotted_spider_mite",
            "Target_Spot",
            "Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato_mosaic_virus",
            "Healthy"
        ]
        model = load_model('model_inception.h5', compile=False)
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        class_idx = np.argmax(preds, axis=1)[0]
        result = class_labels[class_idx] if 0 <= class_idx < len(class_labels) else "Unknown"
        confidence = float(np.max(preds))
        return result, confidence, None

    elif crop_name == 'wheat':
        model = load_model("wheatDiseaseModel.h5", compile=False)
        img = image.load_img(image_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]
        class_label = np.argmax(prediction)
        class_names = ['Leaf rust', 'Loose smut', 'Crown root rot', 'Healthy']
        result = class_names[class_label]
        confidence = float(np.max(prediction))
        return result, confidence, None

    else:
        raise ValueError(f"Unknown crop name: {crop_name}")


class PredictRequest(BaseModel):
    crop_name: str
    image_path: str

@app.get("/predict")
def predict_get(crop_name: str, image_path: str):
    try:
        result, confidence, extra = predict_plant_disease(crop_name, image_path)
        return {"result": result, "confidence": round(confidence * 100, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_post(req: PredictRequest):
    try:
        result, confidence, extra = predict_plant_disease(req.crop_name, req.image_path)
        return {"result": result, "confidence": round(confidence * 100, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-file")
def predict_file(crop_name: str = Form(...), image: UploadFile = File(...) ):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[-1]) as tmp:
            shutil.copyfileobj(image.file, tmp)
            tmp_path = tmp.name
        result, confidence, extra = predict_plant_disease(crop_name, tmp_path)
        os.remove(tmp_path)
        return {"result": result, "confidence": round(confidence * 100, 2)}
    except Exception as e:
        return {"error": str(e)}

"""
Example curl command to test POST endpoint:
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"crop_name": "maize", "image_path": "cor.jpg"}'

Example curl command to test POST file endpoint:
curl -X POST "http://localhost:8000/predict-file" -F crop_name=maize -F image=@cor.jpg
"""
