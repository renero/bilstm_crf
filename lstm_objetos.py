import data
import pandas as pd
import sys
import vocabulary

from tqdm import tqdm
from sys import stdout
from model import Model

training = True

params = data.init()
model = Model()

if training is True:
    datasets = data.prepare(params)
    data.save(datasets, params)
    model.init(params)
    model.train(datasets['train'], datasets['target'], params)
    model.save('output/nn_entities.h5')
else:
    datasets = data.read(params)
    model.load('output/nn_entities.h5')


print('Evaluating test set performance...', flush=True)
frases_test = datasets['utt_test'].reset_index()
hash_test = datasets['hash_test'].reset_index()
amr_test = datasets['amr_test'].reset_index()
pred = []
for sentence in tqdm(datasets['utt_test']['frase'], ascii=True, file=stdout):
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
print('Accuracy: {:.4f}'.format(positives/total_utts))
sys.exit(0)

# --

amr_pred = pred['pred'].apply(lambda x: vocabulary.construccion_amr_objetos(x))
amr_pred = pd.DataFrame(amr_pred)
amr_pred.columns = ['amr_pred']

evaluacion = pd.concat(
    [
        frases_test[['frase']], hash_test[['tag']], amr_test[['amr']], pred,
        amr_pred
    ],
    axis=1)
evaluacion.to_csv("./output/evaluacion.csv", encoding='utf8')

evaluacion = pd.read_csv(
    './output/evaluacion.csv', sep=',', encoding="Latin1")  # TODO
print('Número de documentos de test: ', len(frases_test))
aciertos_tag = evaluacion[evaluacion['tag'] == evaluacion['pred']]
print('   Acierto tag: ', len(aciertos_tag), ' - ',
      round(len(aciertos_tag) / len(frases_test) * 100, 2), '%')

aciertos_amr = evaluacion[(
    (evaluacion['amr'] == evaluacion['amr_pred']) |
    (evaluacion['amr'].isnull() & evaluacion['amr_pred'].isnull()))]
print('   Acierto amr: ', len(aciertos_amr), ' - ',
      round(len(aciertos_amr) / len(frases_test) * 100, 2), '%')

########################################
#
# Pruebas
#
########################################

# model = load_model('./output/BI_LSTM_seguros_objeto.h5')
# largo_max = 40
# sentence = 'viajo a bla bla bla por negrocios, me cubre allí mi seguro'
# sentence = raw_input()
# salida = evaluacion_objeto(sentence, model, params)
# print(salida)
