from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

class NlpPredict():
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def set_sentence(self, sentence):
        test_commentary = []
        test_commentary.append(sentence)
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(test_commentary)
        sequences = tokenizer.texts_to_sequences(test_commentary)
        self.padded = pad_sequences(sequences, maxlen=91, padding='post')

    def predict(self):
        res_predict = self.model.predict(self.padded)
        res_np_float = res_predict.astype(np.float)
        self.res = float(res_np_float[0][0])

        return self.res

# if __name__ == '__main__':
#     x = NlpPredict('nlp_model/sentimentanalysisv4.h5')
#     x.set_sentence("Orang ini baik")

#     r = x.predict()
#     print(r)
#     print(type(r))