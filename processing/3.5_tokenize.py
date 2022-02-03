from tokenizer import Tokenizer
import sys
import pickle

smlfile = './output/dataset.srcml.seq'
smltok_file = 'output/smls.tok'

p = pickle.load(open('output/tdats.tok', 'rb'), encoding='UTF-8')

q = p
q.update_from_file(smlfile)
q.save(smltok_file)
