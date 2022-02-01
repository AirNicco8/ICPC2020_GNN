import tokenizer
import pickle
import sys
import uuid

comlen = 13
sdatlen = 20 # average is 8 functions per file
tdatlen = 100
smllen = 100 # average is 870

def save(obj, filename):
	pickle.dump(obj, open(filename, 'wb'))

coms_trainf = './output/coms.train'
coms_valf = './output/coms.val'
coms_testf = './output/coms.test'
comlen = comlen

tdats_trainf = './output/tdats.train'
tdats_valf = './output/tdats.val'
tdats_testf = './output/tdats.test'


comstok = tokenizer.Tokenizer().load('coms.tok')
smlstok = tokenizer.Tokenizer().load('smls.tok')
tdatstok = smlstok # same tokenizer for smls and tdats so we can share embedding

com_train = comstok.texts_to_sequences_from_file(coms_trainf, maxlen=comlen)
com_val = comstok.texts_to_sequences_from_file(coms_valf, maxlen=comlen)
com_test = comstok.texts_to_sequences_from_file(coms_testf, maxlen=comlen)
tdats_train = tdatstok.texts_to_sequences_from_file(tdats_trainf, maxlen=tdatlen)
tdats_val = tdatstok.texts_to_sequences_from_file(tdats_valf, maxlen=tdatlen)
tdats_test = tdatstok.texts_to_sequences_from_file(tdats_testf, maxlen=tdatlen)

# now split up the srcml asts

srcml_nodes = pickle.load(open('./output/dataset.srcml_nodes.pkl', 'rb'))
srcml_edges = pickle.load(open('./output/dataset.srcml_edges.pkl', 'rb'))

srcml_train_nodes = dict()
srcml_train_edges = dict()
srcml_val_nodes = dict()
srcml_val_edges = dict()
srcml_test_nodes = dict()
srcml_test_edges = dict()

for line in open(tdats_trainf, 'r'):
    (fid, tdat) = line.split(',')
    fid = int(fid)
    srcml_train_nodes[fid] = srcml_nodes[fid]
    srcml_train_edges[fid] = srcml_edges[fid]

for line in open(sdats_valf, 'r'):
    (fid, tdat) = line.split(',')
    fid = int(fid)
    srcml_val_nodes[fid] = srcml_nodes[fid]
    srcml_val_edges[fid] = srcml_edges[fid]

for line in open(sdats_testf, 'r'):
    (fid, tdat) = line.split(',')
    fid = int(fid)
    srcml_test_nodes[fid] = srcml_nodes[fid]
    srcml_test_edges[fid] = srcml_edges[fid]

assert len(com_train) == len(tdats_train)
assert len(com_val) == len(tdats_val)
assert len(com_test) == len(tdats_test)

out_config = {'tdatvocabsize': tdatstok.vocab_size, 'sdatvocabsize': sdatstok.vocab_size, 'comvocabsize': comstok.vocab_size,
            'smlvocabsize': smlstok.vocab_size, 'sdatlen': sdatlen, 'tdatlen': tdatlen, 'comlen': comlen,
            'smllen': smllen}

dataset = {'ctrain': com_train, 'cval': com_val, 'ctest': com_test,
			'dttrain': tdats_train, 'dtval': tdats_val, 'dttest': tdats_test,
			'strain_nodes': srcml_train_nodes, 'sval_nodes': srcml_val_nodes, 'stest_nodes': srcml_test_nodes,
			'strain_edges': srcml_train_edges, 'sval_edges': srcml_val_edges, 'stest_edges': srcml_test_edges,
			'comstok': comstok, 'tdatstok': tdatstok, 'smltok': smlstok,
            'config': out_config}

save(dataset, 'dataset.pkl')
