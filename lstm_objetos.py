import pandas as pd
import data
import pickle
import vocabulary

from keras.models import load_model
from model import Model

params = data.init()
data = data.prepare(params)

# model = Model(params)
# model.train(data['train'], data['target'], params)
# model.save('output/nn_entities.h5')

model = load_model('output/nn_entities.h5')

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
import numpy as np
r = tokenizer.texts_to_sequences(np.array(['durante el mes de julio me desplazo a slough']))
resp = vocabulary.evaluacion_objeto('tengo que pagar franquicia por operacion',
                                    model, params)

pred = []
for sentence in data['utt_test']['frase']:
    traduccion = vocabulary.evaluacion_objeto(sentence, model, params)
    pred.append(traduccion)
pred = pd.DataFrame(pred)
pred.columns = ['pred']
amr_pred = pred['pred'].apply(lambda x: vocabulary.construccion_amr_objetos(x))
amr_pred = pd.DataFrame(amr_pred)
amr_pred.columns = ['amr_pred']
frases_test = data['utt_test'].reset_index()
hash_test = data['hash_test'].reset_index()
amr_test = data['amr_test'].reset_index()
evaluacion = pd.concat(
    [
        frases_test[['frase']], hash_test[['tag']], amr_test[['amr']], pred,
        amr_pred
    ],
    axis=1)
evaluacion.to_csv("./output/evaluacion.csv", encoding='latin1')

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
