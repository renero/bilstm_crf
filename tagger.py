import pickle

from keras.preprocessing.text import Tokenizer
from re import sub, compile
from unidecode import unidecode

# Expresiones regulares para partir la frase
_WORD_SPLIT = compile(r"([.,!?\"';)(])")
_DIGIT_RE = compile(r"\d")


# Función cleanup de texto:
def cleanup(text_string):
    # Quitar acentos:
    text_string = unidecode(text_string)
    # Quitar caracteres extraños
    text_string = sub('[^a-zA-Z0-9ñ:<> ]', '', text_string)
    # Dejar sólo un espacio entre palabras
    text_string = " ".join(text_string.split())
    # Poner todo en minuscula y sin espacios al principio y al final
    return text_string.strip().lower()


def init(dataset, unknown_tag):
    print('Building tokenizer object.')
    tokenizer = Tokenizer(
        lower=True,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        oov_token=unknown_tag)
    tokenizer.fit_on_texts(dataset)
    return tokenizer


def save(tokenizer, tokenizer_name):
    print('Saving tokenizer object ({}).'.format(tokenizer_name))
    with open(tokenizer_name, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Inicialización del universo de palabras
def read(tokenizer_name):
    print('Loading tokenizer object ({})'.format(tokenizer_name))
    with open(tokenizer_name, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def expand_amr(sentence, params):
    amr = ''
    tags = sentence.strip().split()
    valid_tags = params['learn_tags'].copy()
    del valid_tags[params['learn_tags'].index(params['void_tag'])]
    for valid_tag in valid_tags:
        if any(valid_tag in tag for tag in tags):
            amr += '{} '.format(params['amr'][valid_tag])
    return amr.strip()
