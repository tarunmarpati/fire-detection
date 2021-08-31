from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img



unique_labels = ["fire detected.", "no fire detected."]
model = keras.models.load_model("model3.h5")
def predict(img_path):  # mandatory: function name should be predict and it accepts a string which is image location
    img = load_img(img_path, target_size=(256, 256))
    # plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    yhat = model.predict(img)
    yhat = np.array(yhat)
    indices = np.argmax(yhat, axis=1)
    scores = yhat[np.arange(len(yhat)), indices]
    predicted_categories = [unique_labels[i] for i in indices]
    category = predicted_categories[0]
    confidence = round(scores[0] * 100, 2)
    output = category + " (Confidence: " + str(confidence) + "%)"
    return output
