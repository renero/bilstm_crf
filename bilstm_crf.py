"""
bilstm_crf

Usage:
    bilstm_crf train
    bilstm_crf test
    bilstm_crf -h | --help

Options:
  -h, --help    Show this message

"""

import data
import pandas as pd
import sys
import vocabulary

from docopt import docopt
from model import Model
from tqdm import tqdm
from sys import stdout

if __name__ == '__main__':
    arguments = docopt(__doc__, version='FIXME')

params = data.init()
model = Model()
datasets = data.prepare(params)

if arguments['train'] is True:
    model.init(params)
    model.train(datasets['train'], datasets['target'], params)
    model.save('output/nn_entities.h5')
    sys.exit(0)
else:
    model.init(params)
    model.load('output/nn_entities.h5', params)
    print('Evaluating test set performance...', flush=True)

    frases_test = datasets['utt_test'].reset_index()
    hash_test = datasets['hash_test'].reset_index()
    amr_test = datasets['amr_test'].reset_index()
    pred = []
    for sentence in tqdm(
            datasets['utt_test']['frase'], ascii=True, file=stdout):
        traduccion = model.predict(sentence, params)
        pred.append(traduccion)

    pred = pd.DataFrame(pred)
    pred.columns = ['pred']
    total_utts = hash_test.shape[0]
    hash_test = datasets['hash_test'].reset_index()
    positives = 0
    for i in range(total_utts):
        if pred.iloc[i][0] == hash_test.iloc[i]['tag']:
            positives += 1
    print('Accuracy: {:.4f}'.format(positives / total_utts))

# --

amr_pred = pred['pred'].apply(lambda x: vocabulary.expand_amr(x))
amr_pred = pd.DataFrame(amr_pred)
amr_pred.columns = ['amr_pred']

eval = pd.concat(
    [
        frases_test[['frase']], hash_test[['tag']], amr_test[['amr']], pred,
        amr_pred
    ],
    axis=1)
eval.to_csv("./output/eval.csv", encoding='utf8')

eval = pd.read_csv('./output/eval.csv', sep=',', encoding="utf8")  # TODO
print('Nr of test utterances: ', len(frases_test))
aciertos_tag = eval[eval['tag'] == eval['pred']]
print('   Correct classif. TAG: ', len(aciertos_tag), ' - ',
      round(len(aciertos_tag) / len(frases_test) * 100, 2), '%')

aciertos_amr = eval[((eval['amr'] == eval['amr_pred']) |
                     (eval['amr'].isnull() & eval['amr_pred'].isnull()))]
print('   Correct classif. AMR: ', len(aciertos_amr), ' - ',
      round(len(aciertos_amr) / len(frases_test) * 100, 2), '%')
