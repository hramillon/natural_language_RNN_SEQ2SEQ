import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = tf.keras.models.load_model('../gru.keras')
with open('../modelstokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_len = 20

def generate_text_beam_search(seed_text, next_words, beam_width=5, temperature=0.7):
    sequences = [[seed_text.split(), 0.0]]
    
    for _ in range(next_words):
        all_candidates = []
        for seq, score in sequences:
            text_seq = ' '.join(seq)
            token_list = tokenizer.texts_to_sequences([text_seq])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            
            predicted_probs = model.predict(token_list, verbose=0)[0]
            predicted_probs = np.log(predicted_probs + 1e-10) / temperature
            exp_preds = np.exp(predicted_probs - np.max(predicted_probs))
            predicted_probs = exp_preds / np.sum(exp_preds)
            
            top_indices = np.argsort(predicted_probs)[-beam_width:]
            
            for index in top_indices:
                output_word = tokenizer.index_word.get(index, "")
                if output_word:
                    candidate_seq = seq + [output_word]
                    candidate_score = score + np.log(predicted_probs[index] + 1e-10)
                    all_candidates.append((candidate_seq, candidate_score))
        
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return ' '.join(sequences[0][0]) if sequences else seed_text

seed_text = input("EEnter a text: ")
num_words = int(input("Nb of words to generate: "))
result = generate_text_beam_search(seed_text, num_words)
print(result)