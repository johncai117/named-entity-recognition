import os
import re
import codecs
import numpy as np


models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")



def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items) if v[1] > 2}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def get_embedding_dict(filename):
    word_to_embd={}
    count=0
    f=open(filename, "r")

    emb=np.zeros([1,20])
    fl =f.readlines()
    for x in fl:
        x1=x.split(' ')
        emb=np.asarray(x1[1:len(x1)-1])
        count+=1
        emb=np.reshape(emb,(1,len(emb)))
        word_to_embd[x1[0]]=emb.astype(float)
    return word_to_embd
