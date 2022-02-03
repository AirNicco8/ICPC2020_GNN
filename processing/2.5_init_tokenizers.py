from tokenizer import Tokenizer
import sys
import pickle

dim = 100000 # !TODO tune this parameter

def init(a, dim, out):
    tok = Tokenizer()
    tok.train_from_file(a[0], dim)
    tok.update_from_file(a[1])
    tok.update_from_file(a[2])
    tok.save(out)

test_dataset = "output/test_dataset.dats"
train_dataset = "output/train_dataset.dats"
val_dataset = "output/train_dataset.dats"
tmp1 = [train_dataset, val_dataset, test_dataset]

init(tmp1, dim, 'output/tdats.tok')

test_dataset = "output/test_dataset.coms"
train_dataset = "output/train_dataset.coms"
val_dataset = "output/train_dataset.coms"
tmp2 = [train_dataset, val_dataset, test_dataset]

init(tmp2, dim, 'output/coms.tok')
