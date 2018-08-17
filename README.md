# BI-LSTM with CRF (Conditional Random Field)

Implementation of a generic and well-known architecture based on bi-directional LSTM with CRF for sequence tagging.

To correctly count on CRF layer, please refer to
[this page](https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/) to know how to use it from Keras, and [this other one](https://github.com/keras-team/keras-contrib) to install Keras-Contrib package. The package may not work with your version of Keras due to a broken compatibility from Keras 2.2.1 that can be solved by using the recommendations on [this page](https://github.com/ekholabs/keras-contrib/commit/0dac2da8a19f34946448121c6b9c8535bfb22ce2).

All this work is based on the original ideas exposed in [original paper by Buillaume Lample](https://arxiv.org/pdf/1603.01360.pdf).

From using the external CRF layer, there're some issues when saving and loading the model, all of them addressed in [this page](https://github.com/keras-team/keras-contrib/issues/125).

The final intention is to build a system that resembles very much the achieved by the [anago](https://github.com/Hironsan/anago) system which uses bidirectional LSTMs with CRF.

Note: It would be interesting to know more about the NLU capabilities shown in [this repo])https://github.com/SNUDerek/multiLSTM) as it introduces a nice combination of slot_filling and intent detection, together with sequence tagging as in this example:

```
query: i want a first class flight to los angeles
slots:
{'class_type': 'first class', 'toloc.city_name': 'los angeles'}
intent: atis_flight
```
