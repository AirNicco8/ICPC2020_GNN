from tokenizer import Tokenizer
import sys
import pickle

coms_vocab = 10908 # orginally was 44707
tdats_vocab = 75000
sml_vocab = 75000

smlfile = './output/dataset.srcml.seq'
sml_vocab = sml_vocab # sml vocab is small so just pick a big number
smltok_file = 'output/smls.tok'

p = pickle.load(open('output/tdats.tok', 'rb'), encoding='UTF-8')

q = p
q.update_from_file(smlfile)
q.save(smltok_file)
