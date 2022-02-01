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
        global i
        i = 0
        global j
        j = 0

    def parse(self, code):
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
        text = bytes(up, 'utf-8')[node.start_byte:node.end_byte]
        text = text.decode("utf-8")
        return text

    def handle_edge_data(self, data, parent, child):
        self.graph.add_edge(parent, child, text=data)

    def handle_node_data(self, data, count):

        # first, do dats text preprocessing
        data = data.lower().rstrip()
        data = remove_extra_spaces(data)
        data = regexes(data)
        data = const_strings(data)

        # second, create a node if there is text
        if(data != '' '''and len(data)<15'''):
            #for d in data.split(' '): # each word gets its own node
            #    if self.is_not_blank(d):
            #global i
            self.graph.add_node(count, text=data)


    def traverse(self, tree):
        def _traverse(node, p):
            global i
            i=i+1
            if(node.children != []):
                tmp2 = p
                global j
                tmp = i+j
                self.handle_node_data(self.get_data(node), tmp)
                for child in node.children:
                    _traverse(child, tmp)
                    j = j+1
                    self.handle_edge_data(child.type, tmp2, tmp)
                    self.print_node(child, tmp)
            else:
                tmp2 = p
                tmp = i+j
                self.handle_node_data(self.get_data(node), tmp)
                self.handle_edge_data(node.type, tmp2, tmp)
                self.print_node(node, tmp)
        root = tree.root_node
        d = 'root'
        self.graph.add_node(0, text=d)
        _traverse(root, 0)

    def get_graph(self):
        return(self.graph)

def pydecode(unit): # get the graph from a code snippet
    parser = MyASTParser()
    parser.parse(unit)
    return parser.get_graph()

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

    fopn = open(outpath_n, 'wb')
    fope = open(outpath_e, 'wb')

    for fid in good_fid:
        try:
            unit = split[fid]
        except:
            unit = ''

        graph = pydecode(unit)
        c += 1

        lens.append(len(graph.nodes.data()))

        nodes = list(graph.nodes.data())
        try:
            nodes = np.asarray([w2i(x[1]['text']) for x in list(graph.nodes.data())])
            edges = nx.adjacency_matrix(graph)
        except:
            eg = nx.Graph()
            eg.add_node(0)
            nodes = np.asarray([0])
            edges = nx.adjacency_matrix(eg)
            blanks += 1

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

smlstok = pickle.load(open('smls.tok', 'rb'), encoding='UTF-8') # !TODO initialize tokenizer for node data

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

#proc(df_tr, tr_fid, outtr_n, outtr_e)
#proc(df_v, v_fid, outv_n, outv_e)
proc(df_te, te_fid, outte_n, outte_e)
