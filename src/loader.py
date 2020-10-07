import os
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
# from utils import iob2, iob_iobes


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words))
    )
    return dico, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag



def prepare_sentence(str_words, word_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
    }


def prepare_dataset(sentences, mode=None, lower=False, word_to_id=None, tag_to_id=None):
    """
    Prepare the dataset. Return 'data', which is a list of lists of dictionaries containing:
        - words (strings)
        - word indexes
        - tag indexes
    """
    assert mode == 'train' or (mode == 'test' and word_to_id and tag_to_id)

    if mode=='train':
        word_dic, word_to_id, id_to_word = word_mapping(sentences, lower)
        tag_dic, tag_to_id, id_to_tag = tag_mapping(sentences)

    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'tags': tags,
        })

    if mode == 'train':
        return data, {'word_to_id':word_to_id, 'id_to_word':id_to_word, 'tag_to_id':tag_to_id, 'id_to_tag':id_to_tag}
    else:
        return data
