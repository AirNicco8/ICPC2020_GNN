from bs4 import BeautifulSoup
from myutils import prep, drop, print_ast
import multiprocessing
import pickle
import networkx as nx
import pandas as pd
import re
import statistics
import numpy as np
from tree_sitter import Language, Parser

PY_LANGUAGE = Language('../build/my-languages.so', 'python')

df_tr = pd.read_pickle("./dataframes/train_py.pkl")
df_v = pd.read_pickle("./dataframes/val_py.pkl")
df_te = pd.read_pickle("./dataframes/test_py.pkl")

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

re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

class MyASTParser():
    def __init__(self):
        self.graph = nx.Graph()
        self.parser = Parser()
        self.parser.set_language(PY_LANGUAGE)
        global i
        i = 0
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

    def print_node(node):
        text self.get_data(node)
        pos_point = f"[{node.start_point},{node.end_point}]"
        pos_byte = f"({node.start_byte},{node.end_byte})"
        global i
        c = i
        print(
        f"{c:<10}"
        f"{repr(node.type):<25}{'is_named' if node.is_named else '-':<20}"
        f"{pos_point:<30}{pos_byte:<30}"
        f"{text}"
        )

    def handle_data(self, data, parent, count):

        # first, do dats text preprocessing
        data = re_0001_.sub(re_0002, data).lower().rstrip()

        # second, create a node if there is text
        if(data != ''):
            for d in data.split(' '): # each word gets its own node
                if d != '':
                    global i
                    self.graph.add_node(i, text=d)
                    self.graph.add_edge(parent, parent+count)

    def traverse(self, tree):
        def _traverse(node, p):
            global i
            i=i+1
            tmp = i
            if(node.children != []):
                for child in node.children:
                    tmp2 = p
                    _traverse(child, tmp)
                    self.handle_data(self.get_data(child), child.type, tmp2, tmp)
                    self.print_node(child, tmp)
            else:
                    tmp2 = p
                    self.handle_data(self.get_data(node), node.type, tmp2, tmp)
                    self.print_node(node, tmp)
        root = tree.root_node
        d = 'root'
        self.graph.add_node(0, text=d)
        '''for c in root.children:
            co+=1
            self.handle_data(self.get_data(c), c.type, 0, co)
            self.print_node(c, 0+co)'''
        _traverse(root, 0)

    def get_graph(self):
        return(self.graph)

c = 0

def pydecode(unit):
    parser = MyASTParser()
    parser.parse(unit)
    return parser.get_graph()

prep('loading tokenizer... ')
smlstok = pickle.load(open('smls.tok', 'rb'), encoding='UTF-8')
drop()

lens = list()
good_fid = load_good_fid()
print('num good fids:', len(good_fid))
srcml_nodes = dict()
srcml_edges = dict()
fopn = open('./output/dataset.srcml_nodes.pkl', 'wb')
fope = open('./output/dataset.srcml_edges.pkl', 'wb')
blanks = 0

def w2i(word):
    try:
        i = smlstok.w2i[word]
    except KeyError:
        i = smlstok.oov_index
    return i

prep('parsing xml... ')
for fid in good_fid:
    try:
        unit = srcmlunits[fid]
    except:
        unit = ''

    (graph, seq) = xmldecode(unit)
    seq = ' '.join(seq)
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
drop()

print('blanks:', blanks)
print('avg:', sum(lens) / len(lens))
print('max:', max(lens))
print('median:', statistics.median(lens))
print('% abv 200:', sum(i > 200 for i in lens) / len(lens))

prep('writing pkl... ')
pickle.dump(srcml_nodes, fopn)
pickle.dump(srcml_edges, fope)
drop()

prep('cleaning up... ')
fopn.close()
fope.close()
drop()
