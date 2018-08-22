import numpy as np
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from re import sub, compile
from unidecode import unidecode


class Tagger:
    # Expresiones regulares para partir la frase
    _WORD_SPLIT = compile(r"([.,!?\"';)(])")
    _DIGIT_RE = compile(r"\d")
    tokenizer = None

    def __init__(self):
        pass

    def init(self, data):
        print('Building tokenizer object.')
        dataset = data.sets['utt_train']['frase']
        unknown_tag = data.params['UNK']
        self.tokenizer = Tokenizer(
            lower=True,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            oov_token=unknown_tag)
        self.tokenizer.fit_on_texts(dataset)
        print('Encoding training sets.', flush=True)
        data.sets['train'] = self.encode_utterances(data, 'utt_train')
        data.sets['target'] = self.encode_prediction(data, 'hash_train')

    def encode_utterances(self, data, dataset_name):
        """Simplified, Keras-way, creation of vocabulary, saving it
        padding and encoding."""
        encoded_utts = self.tokenizer.texts_to_sequences(
            data.sets[dataset_name]['frase'])
        padded_encodings = pad_sequences(
            encoded_utts, maxlen=data.params['max_utt_len'], padding='post')
        data.params['vocabulary_size'] = len(self.tokenizer.word_index) + 1
        data.sets['vocabulary'] = np.array(
            list(self.tokenizer.index_word.values()))
        return padded_encodings

    def encode_prediction(self, data, dataset_name):
        """Builds the responde in a typical supervised learning problem
        from the input dataset."""
        data.sets[dataset_name]['tag'] = data.sets[dataset_name]['tag'].apply(
            lambda x: cleanup(x))
        target = []
        for i, line in enumerate(data.sets[dataset_name]['tag']):
            tags_array = line.strip().split()
            target_line = np.zeros(
                [data.params['max_utt_len'], len(data.params['learn_tags'])],
                dtype=np.int8)
            for j, tag in enumerate(tags_array):
                one_pos = data.params['learn_tags'].index(tag)
                target_line[j][one_pos] = 1
            target.append(target_line)
        return np.array(target)

    def save(self, tokenizer_name):
        print('Saving tokenizer object ({}).'.format(tokenizer_name))
        with open(tokenizer_name, 'wb') as handle:
            pickle.dump(
                self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Inicialización del universo de palabras
    def load(self, tokenizer_name):
        print('Loading tokenizer object ({})'.format(tokenizer_name))
        with open(tokenizer_name, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        return self


def cleanup(text_string):
    # Quitar acentos:
    text_string = unidecode(text_string)
    # Quitar caracteres extraños
    text_string = sub('[^a-zA-Z0-9ñ:<> ]', '', text_string)
    # Dejar sólo un espacio entre palabras
    text_string = " ".join(text_string.split())
    # Poner todo en minuscula y sin espacios al principio y al final
    return text_string.strip().lower()


def expand_amr(sentence, params):
    amr = ''
    tags = sentence.strip().split()
    valid_tags = params['learn_tags'].copy()
    del valid_tags[params['learn_tags'].index(params['void_tag'])]
    for valid_tag in valid_tags:
        if any(valid_tag in tag for tag in tags):
            amr += '{} '.format(params['amr'][valid_tag])
    return amr.strip()
