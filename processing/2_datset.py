import pickle
import re
import collections
import sys
import pandas as pd

# PROCESSING OF CODE AS TEXT

re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

# load data to preprocess
df_tr = pd.read_pickle("./dataframes/train_py.pkl")
df_v = pd.read_pickle("./dataframes/val_py.pkl")
df_te = pd.read_pickle("./dataframes/test_py.pkl")

def load(filename):
    return pickle.load(open(filename, 'rb'))

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

def delete_comment(s): #deletes the comment from the function
    s = re.sub(r'(#.*)', '', s)
    s= re.sub(r'(\'\'\')[\s\S]*?(\'\'\')', "", s, re.S)
    s= re.sub(r'(\"\"\")[\s\S]*?(\"\"\")', "", s, re.S)
    return s

def proc(path, ids, df): # process the text code and dump it in a datafile
    fo = open(path, 'w')
    for i in ids:
        code = df[i]
        code = delete_comment(code)
        code = re_0001_.sub(re_0002, code)

        if code == '':
            continue
        code = code.split()
        code = ' '.join(code)

        fo.write('{}, <s> {} </s>\n'.format(i, code))

    fo.close()

outfile1 = './output/train_dataset.dats'
outfile2 = './output/val_dataset.dats'
outfile3 = './output/test_dataset.dats'

df_tr = df_tr['code']
trids = df_tr.index
df_v = df_v['code']
vids = df_v.index
df_te = df_te['code']
tsids = df_te.index

proc(outfile1, trids, df_tr)
proc(outfile2, vids, df_v)
proc(outfile3, tsids, df_te)
