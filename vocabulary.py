import numpy as np
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


def save(vocab_list, params):
    if len(vocab_list) > params['max_vocabulary_size']:
        vocab_list = vocab_list[:params['max_vocabulary_size']]
    if 'vocabulary_path' not in params:
        vocabulary_path = join(
            join(params['working_path'], 'output'),
            params['output_vocabulary'])
        params['vocabulary_path'] = vocabulary_path
    with tf.gfile.GFile(params['vocabulary_path'], mode="wb") as vocab_file:
        for w in vocab_list:
            vocab_file.write(w + b"\n")
    print('Vocabulary saved to', params['vocabulary_path'])


# Inicialización del universo de palabras
def read(params):
    print("Reading vocabulary...", flush=True)
    if tf.gfile.Exists(params['vocabulary_path']):
        rev_vocab = []
        with tf.gfile.GFile(params['vocabulary_path'], mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        print('done.', flush=True)
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.",
                         params['vocabulary_path'])


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


def evaluacion_objeto(sentence, model, params):
    # Tratamieto
    tratada = cleanup(sentence)
    vocab, _ = read(params)

    # Tokenizado
    token_sentence = encode_sentence(
        tratada, vocab, tokenizer=None, normalize_digits=False)
    largo_real_frase = len(token_sentence)
    while len(token_sentence) < params['largo_max']:
        token_sentence.append(0)
    test = []
    test += [token_sentence]
    test = np.asarray(test)

    # Predicción
    predict = model.predict(test)
    traduccion = []
    for i in range(0, largo_real_frase):  # TODO
        if np.argmax(predict[0][i]) == 0:
            traduccion += ['x']
        if np.argmax(predict[0][i]) == 1:
            traduccion += ['ot']
        if np.argmax(predict[0][i]) == 2:
            traduccion += ['oc']
        if np.argmax(predict[0][i]) == 3:
            traduccion += ['oh']
        if np.argmax(predict[0][i]) == 4:
            traduccion += ['or']
        if np.argmax(predict[0][i]) == 5:
            traduccion += ['os']

    # Escritura en fichero:
    traduccion = " ".join(str(x) for x in traduccion)
    return traduccion
