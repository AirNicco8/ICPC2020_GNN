import pickle
import pandas as pd
import re

# PROCESSING OF COMMENTS

def remove_URL(text):
    """Remove URLs from a text string"""
    return re.sub(r"http\S+", "", text)

def remove_nonascii(text):
    return re.sub(r"[^a-zA-Z0-9/._ ]+", "",text)

def clean(text): # clean a given string
    text = remove_URL(text)
    text = remove_nonascii(text)
    return text

def proc(path, ids, df): # clean the diven dataframes and write the output as datafile
    fo = open(path, 'w')
    g_ids=[]
    for i in ids:
        com = df[i]
        com = clean(com)

        if com == '':
            continue
        com = com.split()
        if len(com) > 13 or len(com) < 2: # discard comments which are too long or too short
        	continue
        if ('Auto' in com) and ('Generated' in com) and ('Code' in com): # discard auto Generated comments
            continue
        com = ' '.join(com)
        g_ids.append(i)
        fo.write('{}, <s> {} </s>\n'.format(i, com))

    fo.close()
    return g_ids

# loading of data to process
df_tr = pd.read_pickle('./dataframes/train_py.pkl')
df_v = pd.read_pickle("dataframes/val_py.pkl")
df_te = pd.read_pickle("dataframes/test_py.pkl")

troutfile = './output/train_dataset.coms'
valoutfile = './output/val_dataset.coms'
tstoutfile = './output/test_dataset.coms'

#extract the indexes
df_tr = df_tr['docstring']
trids = df_tr.index
df_v = df_v['docstring']
vids = df_v.index
df_te = df_te['docstring']
tsids = df_te.index

# process the data
g_tr = proc(troutfile, trids, df_tr)
g_v = proc(valoutfile, vids, df_v)
g_te = proc(tstoutfile, tsids, df_te)

with open('output/good_ids_tr.pkl', 'wb') as f:
    pickle.dump(g_tr, f)
f.close()

with open('output/good_ids_v.pkl', 'wb') as f:
    pickle.dump(g_v, f)
f.close()

with open('output/good_ids_te.pkl', 'wb') as f:
    pickle.dump(g_te, f)
f.close()
