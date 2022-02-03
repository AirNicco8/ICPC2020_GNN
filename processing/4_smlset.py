import multiprocessing
import pickle
import networkx as nx
import pandas as pd
import re
import statistics
import numpy as np
from tree_sitter import Language, Parser

PY_LANGUAGE = Language('../build/my-languages.so', 'python')

# load dataset to process
df_tr = pd.read_pickle("./dataframes/train_py.pkl")
df_v = pd.read_pickle("./dataframes/val_py.pkl")
df_te = pd.read_pickle("./dataframes/test_py.pkl")

df_tr = df_tr['code']
df_v = df_v['code']
df_te = df_te['code']

def remove_extra_spaces(text): # reduces more than one space to 1 in graph data
    return re.sub(r"[ ]+", " ",text)

stringc = r'"([A-Za-z0-9_\./\\-]*)"|\'([A-Za-z0-9_\./\\-]*)\''
def const_strings(text): # replace constant string assignment with unique text
    return re.sub(stringc, "string", text)

regex =  r'r\'([A-Za-z0-9_\./\\+*\-x^\[\]\(\)]*)\''
def regexes(text): #remove regexes and replace with unique text
    return re.sub(regex, "regex", text)

def load_good_fid(split): # extract indexes from dataframe
    good_fid = []
    ids = split.index
    good_fid = list(ids)
    return good_fid

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp) # data cleaning

re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])') # more cleaning

class MyASTParser(): # this class parse python code - using ASTs - to extract graphs
    def __init__(self):
        self.graph = nx.Graph()
        self.parser = Parser()
        self.parser.set_language(PY_LANGUAGE)
        self.code = ''
        self.i = 0
        self.seq = list()

    def parse(self, code):
        code = self.delete_comment(code)
        self.code = code
        tree = self.parser.parse(bytes(code, "utf8"))
        self.traverse(tree)

    def is_not_blank(self, s):
        return bool(s and not s.isspace())

    def delete_comment(self, s):
        s = re.sub(r'(#.*)', '', s)
        s= re.sub(r'(\'\'\')[\s\S]*?(\'\'\')', "", s, re.S)
        s= re.sub(r'(\"\"\")[\s\S]*?(\"\"\")', "", s, re.S)
        return s

    def get_data(self,node):
        text = bytes(self.code, 'utf-8')[node.start_byte:node.end_byte]
        text = text.decode("utf-8")
        return text

    def handle_data(self, data, parent):
    # first, do dats text preprocessing
        data = re_0001_.sub(re_0002, data).lower().rstrip()
        data = remove_extra_spaces(data)
        data = regexes(data)
        data = const_strings(data)

            # second, create a node if there is text
        if(self.is_not_blank(data)):
            for d in data.split(' '): # each word gets its own node
                if self.is_not_blank(d):
                    self.i = self.i+1
                    self.seq.append(d)
                    self.graph.add_node(self.i, text=d)
                    self.graph.add_edge(parent, self.i)

    def traverse(self, tree):
        def _traverse(node, p):
            self.i = self.i+1
            self.seq.append(node.type)
            self.graph.add_node(self.i, text=node.type)
            self.graph.add_edge(p, self.i)
            tmp = self.i
            self.handle_data(self.get_data(node), self.i)
            for child in node.children:
                _traverse(child, tmp)

                #self.print_node(child, self.i)
        root = tree.root_node
        self.graph.add_node(0, text='root')
        _traverse(root, 0)

    def get_graph(self):
        return(self.graph)

    def get_seq(self):
        return(self.seq)

def pydecode(unit): # get the graph from a code snippet
    parser = MyASTParser()
    parser.parse(unit)
    return(parser.get_graph(), parser.get_seq())

def w2i(word):
    try:
        i = smlstok.w2i[word]
    except KeyError:
        i = smlstok.oov_index
    return i

def proc(split, good_fid, outpath_n, outpath_e): # given the dataframe to process extract graph features to dicts and dump it into a pickle
    c = 0
    blanks = 0
    srcml_nodes = dict()
    srcml_edges = dict()
    #print('processing file %s' % (split))
    fopn = open(outpath_n, 'wb')
    fope = open(outpath_e, 'wb')

    for fid in good_fid:
        try:
            unit = split[fid]
        except:
            unit = ''

        (graph, seq) = pydecode(unit)

        c += 1

        lens.append(len(graph.nodes.data()))

        nodes = list(graph.nodes.data())

        #print(nodes)
        #print('%'*80)
        #print([w2i(x[1]['text']) for x in list(graph.nodes.data())])
        #print(nx.adjacency_matrix(graph))
        try:
            nodes = np.asarray([w2i(x[1]['text']) for x in list(graph.nodes.data())])
            edges = nx.adjacency_matrix(graph)
        except:
            eg = nx.Graph()
            eg.add_node(0)
            nodes = np.asarray([0])
            edges = nx.adjacency_matrix(eg)
            blanks += 1
        #print(nodes)
        srcml_nodes[int(fid)] = nodes
        srcml_edges[int(fid)] = edges

        if(c % 10000 == 0):
            print(c)

    print('blanks:', blanks)
    print('avg:', sum(lens) / len(lens))
    print('max:', max(lens))
    print('median:', statistics.median(lens))
    print('% abv 200:', sum(i > 200 for i in lens) / len(lens))

    pickle.dump(srcml_nodes, fopn)
    pickle.dump(srcml_edges, fope)

    fopn.close()
    fope.close()

smlstok = pickle.load(open('output/smls.tok', 'rb'), encoding='UTF-8') # !TODO initialize tokenizer for node data

# here we actually process the data with the functions above

lens = list()
tr_fid = load_good_fid(df_tr)
v_fid = load_good_fid(df_v)
te_fid = load_good_fid(df_te)

outtr_n = './output/dataset.tr_nodes.pkl'
outtr_e = './output/dataset.tr_edges.pkl'
outv_n = './output/dataset.v_nodes.pkl'
outv_e = './output/dataset.v_edges.pkl'
outte_n = './output/dataset.te_nodes.pkl'
outte_e = './output/dataset.te_edges.pkl'

proc(df_tr, tr_fid, outtr_n, outtr_e)
proc(df_v, v_fid, outv_n, outv_e)
proc(df_te, te_fid, outte_n, outte_e)
