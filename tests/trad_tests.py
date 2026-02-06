import numpy as np
import pandas as pd
import re
import sentencepiece as spm
from tensorflow import keras
from sklearn.model_selection import train_test_split

model = keras.models.load_model('../models/final_translation2.h5')

df = pd.read_csv(
    "../ressources/trad_max_cleaned.csv",
    sep="\t",
    header=None,
    names=["fr", "ang"]
)

def add_spaces_around_punctuation(text):
    text = re.sub(r'\s+([.,?!";()\[\]\-$€£])', r'\1', text)
    text = re.sub(r'([.,?!";()\[\]\-$€£])\s*', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["fr"] = df["fr"].apply(add_spaces_around_punctuation)
df["ang"] = df["ang"].apply(add_spaces_around_punctuation)

df_train, df_tmp = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_tmp, test_size=0.5, random_state=42)

sp_fr = spm.SentencePieceProcessor(model_file="../models/sp_fr.model")
sp_en = spm.SentencePieceProcessor(model_file="../models/sp_en.model")

vocab_en = sp_en.vocab_size()

lengths_fr = [len(sp_fr.EncodeAsIds(text)) + 1 for text in df_train["fr"].tolist()]
lengths_en = [len(sp_en.EncodeAsIds(text)) + 2 for text in df_train["ang"].tolist()]

max_len_fr = int(np.percentile(lengths_fr, 99))
max_len_en = int(np.percentile(lengths_en, 99))

print(f"Max lengths: FR={max_len_fr}, EN={max_len_en}\n")

def pad_sequence(seq, maxlen):
    if len(seq) < maxlen:
        return seq + [0] * (maxlen - len(seq))
    return seq[:maxlen]

def translate_greedy(sentence):
    sentence_preprocessed = add_spaces_around_punctuation(sentence.lower())
    tokens_fr = sp_fr.EncodeAsIds(sentence_preprocessed)
    tokens_fr = tokens_fr + [2]
    encoder_input = np.array([pad_sequence(tokens_fr, max_len_fr)])
    
    decoder_input = np.zeros((1, max_len_en - 1))
    decoder_input[0, 0] = 1
    
    translation_tokens = []
    
    for step in range(max_len_en - 1):
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        predicted_id = np.argmax(predictions[0, step, :])
        
        if predicted_id == 0 or predicted_id == 2:
            break
        
        translation_tokens.append(int(predicted_id))
        
        if step + 1 < max_len_en - 1:
            decoder_input[0, step + 1] = predicted_id
        
        if len(translation_tokens) > 50:
            break
    
    translation_tokens = [int(t) for t in translation_tokens]
    translation_text = sp_en.DecodeIds(translation_tokens)
    return translation_text

test_sentences = [
    "Je suis fatigué .",
    "il aime les chats .",
    "Il m'a aidé à porter la chaise .",
    "Personne n'a l'air content .",
    "comment allez vous ?",
    "je suis français .",
    "quel est votre nom ?",
    "mon voisin à un beau jardin et une belle voiture .",
    "pour faire de la purée, il faut des patates !",
    df_train["fr"].iloc[1],
    df_train["fr"].iloc[10],
    df_train["fr"].iloc[100]
]

for idx, sent in enumerate(test_sentences, 1):
    pred = translate_greedy(sent)
    print(f"{idx}. FR: {sent}")
    print(f"   EN: {pred}\n")