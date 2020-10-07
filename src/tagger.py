#!/usr/bin/env python

import os
import time
import codecs
import argparse
import json
from tqdm import tqdm
import numpy as np
from loader import prepare_sentence, load_sentences, prepare_dataset
from model import HMM,FFN
import collections
from sklearn.metrics import f1_score, confusion_matrix
from utils import get_embedding_dict

feature_vocab = {}

start2 = time.time()


def tag_corpus(model, test_corpus, output_file, dic, args):
    if output_file:
        f_output = codecs.open(output_file, 'w', 'utf-8')
    start = time.time()

    num_correct = 0.
    num_total = 0.
    y_pred=[]
    y_actual=[]
    print('Tagging...')
    for i, sentence in enumerate(tqdm(test_corpus)):
        tags = model.tag(sentence['words'])
        str_tags = [dic['id_to_tag'][t] for t in tags]
        y_pred.extend(tags)
        y_actual.extend(sentence['tags'])

        # Check accuracy.
        num_correct += np.sum(np.array(tags) == np.array(sentence['tags']))
        num_total += len([w for w in sentence['words']])

        if output_file:
            f_output.write('%s\n' % ' '.join('%s%s%s' % (w, args.delimiter, y)
                                             for w, y in zip(sentence['str_words'], str_tags)))



    print('---- %i lines tagged in %.4fs ----' % (len(test_corpus), time.time() - start))
    if output_file:
        f_output.close()

    print("Overall accuracy: %s\n" % (num_correct/num_total))
    return y_pred,y_actual


def confusionmatrix(y_pred,y_actual):
    A = confusion_matrix(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred,average=None)
    print("Confusion Matrix:\n", A)
    print("F-1 scores: ", f1)


def main(args):
    # get data

    args.lower = False # Do not change this
    args.zeros = True  # Do not change this

    train_sentences = load_sentences(args.train_file, args.lower, args.zeros)
    train_corpus, dic = prepare_dataset(train_sentences, mode='train', lower=args.lower, word_to_id=None, tag_to_id=None)


    print(args.model)

    # train model
    if args.model == 'HMM':
        model = HMM(dic, decode_type=args.decode_type)
        model.train(train_corpus)

    elif args.model == 'FFN':
        embeddings = get_embedding_dict(args.embedding_file)
        model = FFN(dic, embeddings)
        model.train(train_corpus)
    else:
        raise ValueError("Unknown model type")

    print("Train results:")
    pred,real = tag_corpus(model, train_corpus, None, dic, args)

    print("Tags: ", dic['id_to_tag'])
    A = confusionmatrix(pred,real)

    # test on validation
    test_sentences = load_sentences(args.test_file, args.lower, args.zeros)
    test_corpus = prepare_dataset(test_sentences, mode='test', lower=args.lower, word_to_id=dic['word_to_id'], tag_to_id=dic['tag_to_id'])

    print("\n-----------\nTest results:")
    pred,real=tag_corpus(model, test_corpus, args.output_file, dic, args)

    print("Tags: ", dic['id_to_tag'])
    A=confusionmatrix(pred,real)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HMM", help="Type of sequence model",
        choices=['HMM', 'MEMM','FFN'])
    parser.add_argument("--train_file", default="", help="Train file")
    parser.add_argument("--test_file", default="", help="Test file")
    parser.add_argument("--embedding_file", default="vectors_wiki.20.txt", help="Word embeddings file")
    parser.add_argument("--output_file", default="",help="Output file location for test predictions")
    parser.add_argument("--delimiter", default="__",
        help="Delimiter to separate words from their tags")
    parser.add_argument("--lower", default=False, type=bool, help="Convert to lowercase or not")
    parser.add_argument("--zeros", default=False, type=bool, help="Convert numbers to zeros")
    parser.add_argument("--decode_type", default="greedy", help="Type of decoding",
        choices=['viterbi', 'greedy'])

    args = parser.parse_args()

    # Check parameters validity
    assert args.delimiter
    assert os.path.isfile(args.train_file)
    assert os.path.isfile(args.test_file)
    main(args)
    #print(time.time() - start2)
