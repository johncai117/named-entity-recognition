# Named Entity Recognition
Named Entity Recognition for Natural Language Processing

In this project, I investigate the different performance of Hidden Markov Models (HMM) Viterbi decoding and Feedforward Neural Networks in Named Entity Recognition (NER). NER aims to be able to classify words into "named entities", which are usually proper nouns, versus words that are not "named entities".

I find that the HMM Viterbi decoding works better compared to a simple rolling window Feedforward Neural Network.

### Validation accuracies:

HMM Viterbi: 0.92815

Feedforward Neural Network: 0.90113

### Discussion
The results are most likely because of the following limitations of the FNN with rolling window.

A key limitation of feedforward neural networks is the fixed window, which limits the input information into the classifier. The fixed window size means that for sentences that are longer than the window size, the feedforward neural network will not be able to use information from the entire sentence to predict the class. This may be problematic because the information that is arbitrarily distant from the point can affect the process. Thus, with more local information, I would expect the FNN to predict the null class often (in order to minimize the cross-entropy loss, as the null class is the most populous).

### Future Work 
In future work, I am keen to explore how a SOTA transformer architecture or LSTM would perform against the HMM Viterbi baseline.

