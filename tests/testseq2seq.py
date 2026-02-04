import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

# ======================
# PARAMÈTRES (vérifier qu'ils matchent l'entraînement)
# ======================
MODEL_PATH = "../models/seq2seq_fr_en.keras"
TOK_FR_PATH = "models/tokenizer_fr.pkl"
TOK_EN_PATH = "models/tokenizer_en.pkl"

MAX_LEN_FR = 99   # longueur des séquences encoder à l'entraînement
MAX_LEN_EN = 99   # longueur des séquences decoder à l'entraînement

# ======================
# CHARGEMENT DU MODELE ET TOKENIZERS
# ======================
model = keras.models.load_model(MODEL_PATH)

with open(TOK_FR_PATH, "rb") as f:
    tokenizer_fr = pickle.load(f)

with open(TOK_EN_PATH, "rb") as f:
    tokenizer_en = pickle.load(f)

index_to_word_en = {v: k for k, v in tokenizer_en.word_index.items()}

START_TOKEN = tokenizer_en.word_index["<start>"]
END_TOKEN = tokenizer_en.word_index["<end>"]

# ======================
# PRÉTRAITEMENT FRANÇAIS
# ======================
def encode_fr(sentence):
    seq = tokenizer_fr.texts_to_sequences([sentence.lower()])[0]
    seq = seq[:MAX_LEN_FR]  # tronquer si plus long
    seq = keras.preprocessing.sequence.pad_sequences(
        [seq], maxlen=MAX_LEN_FR, padding="post"
    )
    return seq

# ======================
# TRADUCTION (GREEDY DECODING)
# ======================
def translate(sentence):
    enc_input = encode_fr(sentence)

    dec_input = np.zeros((1, MAX_LEN_EN), dtype=np.int32)
    dec_input[0, 0] = START_TOKEN

    result = []

    for t in range(1, MAX_LEN_EN):
        preds = model.predict([enc_input, dec_input], verbose=0)
        token = np.argmax(preds[0, t - 1])

        if token == END_TOKEN:
            break

        result.append(index_to_word_en.get(token, "<unk>"))
        dec_input[0, t] = token

    return " ".join(result)

# ======================
# PHRASES DE TEST
# ======================
tests = [
    "je m'appelle",
    "je suis étudiant",
    "le lit est rouge"
]

for s in tests:
    print(f"FR : {s}")
    print(f"EN : {translate(s)}")
    print("-" * 40)
