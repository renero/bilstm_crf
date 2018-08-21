import pickle

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


def save(tokenizer, params):
    print('Loading tokenizer object used to train the net...')
    with open(params['def_tokenizer_name'], 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Inicialización del universo de palabras
def read(tokenizer_name):
    print('Loading tokenizer object used to train the net...')
    with open(tokenizer_name, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


# Construcción AMR objetos
def expand_amr(sentence):  # TODO
    amr = ''
    # Objetos:
    if sentence.find("ot") != -1:
        amr = amr + "<o:tarjeta>"
    if sentence.find("oc") != -1:
        amr = amr + "<o:coberturas>"
    if sentence.find("oh") != -1:
        amr = amr + "<o:centros>"
    if sentence.find("os") != -1:
        amr = amr + "<o:seguro>"
    if sentence.find("or") != -1:
        amr = amr + "<o:reembolso>"

    return amr
