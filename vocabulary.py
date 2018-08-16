import numpy as np
import pickle
import tensorflow as tf

from os.path import join
from re import sub, compile
from sys import stdout
from tqdm import tqdm
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


# Parseador
def basic_tokenizer(sentence, binary=False):
    words = []
    for space_separated_fragment in sentence.strip().split():
        if binary:
            space_separated_fragment = space_separated_fragment.decode(
                'utf-8', 'ignore')
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def save(tokenizer, params):
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Inicialización del universo de palabras
def read(params):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def create(infile, tokenizer=None, normalize_digits=False, params=None):
    """Creates the dictionary with the infile passed as 1st argument and saves
    it to a file"""
    print("Creating vocabulary...", flush=True)
    vocab = {}
    for line in tqdm(infile['frase'], ascii=True, file=stdout):
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(
            line, binary=True)
        for w in tokens:
            word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab_list = params['_START_VOCAB'] + sorted(
        vocab, key=vocab.get, reverse=True)
    save(vocab_list, params)
    return vocab


# Conversión de una frase a enteros
def encode_sentence(sentence,
                    vocabulary,
                    tokenizer=None,
                    normalize_digits=False,
                    params=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, params['UNK_ID']) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [
        vocabulary.get(_DIGIT_RE.sub(b"0", w), params['UNK_ID']) for w in words
    ]


# Conversión de varias frases a enteros
def encode(dataset_name,
           datasets,
           tokenizer=None,
           normalize_digits=False,
           params=None):
    ids = []
    print('Encoding sentences with vocabulary...', flush=True)
    if 'vocabulary' not in datasets:
        params['vocabulary'], _ = read(params)
    for line in tqdm(datasets[dataset_name]['frase'], ascii=True, file=stdout):
        token_ids = encode_sentence(line, datasets['vocabulary'], tokenizer,
                                    normalize_digits, params)
        while len(token_ids) < params['largo_max']:
            token_ids.append(0)
        ids.append(token_ids)
    return ids


# Construcción AMR objetos
def construccion_amr_objetos(sentence):  # TODO
    amr = ''
    # Objetos:
    if sentence.find("ot") != -1:
        amr = amr + "<o:tarjeta> "
    if sentence.find("oc") != -1:
        amr = amr + "<o:coberturas> "
    if sentence.find("oh") != -1:
        amr = amr + "<o:centros> "
    if sentence.find("os") != -1:
        amr = amr + "<o:seguro> "
    if sentence.find("or") != -1:
        amr = amr + "<o:reembolso> "

    return amr
