import pickle
import pandas as pd
import re

def remove_URL(text):
    """Remove URLs from a text string"""
    return re.sub(r"http\S+", "", text)

def remove_nonascii(text):
    return re.sub(r"[^a-zA-Z0-9/._ ]+", "",text)

def clean(text):
    text = remove_URL(text)
    text = remove_nonascii(text)
    return text

def proc(path, ids, df):
    fo = open(path, 'w')
    for i in ids:
        com = df[i]
        com = clean(com)

        if com == '':
            continue
        com = com.split()
        if len(com) > 13 or len(com) < 2:
        	continue
        if ('Auto' in com) and ('Generated' in com) and ('Code' in com):
            continue
        com = ' '.join(com)
        fo.write('{}, <s> {} </s>\n'.format(i, com))

    fo.close()

df_tr = pd.read_pickle('./dataframes/train_py.pkl')
df_v = pd.read_pickle("dataframes/val_py.pkl")
df_te = pd.read_pickle("dataframes/test_py.pkl")

troutfile = './output/train_dataset.coms'
valoutfile = './output/val_dataset.coms'
tstoutfile = './output/test_dataset.coms'

df_tr = df_tr['docstring']
trids = df_tr.index
df_v = df_v['docstring']
vids = df_v.index
df_te = df_te['docstring']
tsids = df_te.index

print(df_tr[2781])

proc(troutfile, trids, df_tr)
proc(valoutfile, vids, df_v)
proc(tstoutfile, tsids, df_te)
