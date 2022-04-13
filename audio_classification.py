import librosa
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tensorflow.keras.models import load_model


def audio_classification(audio_file):
    model2 = load_model("audio_classification2.hdf5")
    df=pd.read_csv("extracted_features.csv")
    y = np.array(df['class'].tolist())
    labelencoder = LabelEncoder()
    y1 = to_categorical(labelencoder.fit_transform(y))
    audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    predicted_label = np.argmax(model2.predict(mfccs_scaled_features), axis=-1)

    prediction_class = labelencoder.inverse_transform(predicted_label)
    result = prediction_class

    return result


