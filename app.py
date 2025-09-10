import gradio as gr
import numpy as np
import cv2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Load and preprocess dataset
digits = load_digits()
X = digits.data
y = digits.target

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train models
models = {
    "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(probability=True, kernel='rbf', gamma='scale', random_state=42)
}

accuracies = {}
recalls = {}
conf_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    accuracies[name] = acc
    recalls[name] = rec
    conf_matrices[name] = cm

# Improved Preprocessing
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) > 127:
        gray = 255 - gray
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    cords = cv2.findNonZero(thresh)

    if cords is None:
        return np.zeros((8, 8))

    x, y, w, h = cv2.boundingRect(cords)
    cropped = thresh[y:y+h, x:x+w]
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    small = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)
    small = (small / 255.0) * 16.0
    return small

# Prediction function
def predict_digit(data, model_choice):
    if data is None or "composite" not in data:
        return {str(i): 0.0 for i in range(10)}

    img = data["composite"]
    arr = preprocess_image(img)
    arr = arr.flatten().reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    if "MLP" in model_choice:
        model = models["MLP"]
    elif "RandomForest" in model_choice:
        model = models["RandomForest"]
    elif "KNN" in model_choice:
        model = models["KNN"]
    else:
        model = models["SVM"]

    probs = model.predict_proba(arr_scaled)[0]
    return {str(i): float(probs[i]) for i in range(10)}

# Gradio Interface
canvas = gr.ImageEditor(
    type="numpy",
    width=200,
    height=200,
    label="Draw the digit here"
)

model_dropdown = gr.Dropdown(
    choices=[
        f"MLP ({accuracies['MLP']:.3f})",
        f"RandomForest ({accuracies['RandomForest']:.3f})",
        f"KNN ({accuracies['KNN']:.3f})",
        f"SVM ({accuracies['SVM']:.3f})"
    ],
    value=f"MLP ({accuracies['MLP']:.3f})",
    label="Choose Model"
)

iface = gr.Interface(
    fn=predict_digit,
    inputs=[canvas, model_dropdown],
    outputs=gr.Label(num_top_classes=10),
    title="Handwritten Digit Recognition (Improved)",
    description="Draw a digit (0-9) and choose a model to classify it. Improved preprocessing for better accuracy."
)

iface.launch()
