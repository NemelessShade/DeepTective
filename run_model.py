from phply.phpparse import make_parser
from phply import phpast as ast
from phply import phplex
from data.preprocessing import sub_tokens
from data.wirecaml_utils.my_php_listener import MyPHPListener
from data.wirecaml_utils.phptraverser import php_traverser
import intervaltree
import os
import re
import pickle
import torch
from model import PhpNetGraphTokensCombine
from data.preprocessing import sub_tokens
from data.preprocessing import map_tokens
from torch_geometric.data import Data
import main
import numpy as np
from torch_geometric.data import DataLoader, DataListLoader, Batch
import random
import pandas as pd

import sys

#data_ls = []
#no_flaws = 0
#parser = make_parser()
#lexer = phplex.lexer.clone()
ignore_tokens = ['LBRACE','RBRACE']

ignore = {'WHITESPACE', 'OPEN_TAG', 'CLOSE_TAG'}

def process(line, allfuncs,allvars):
    lexer = phplex.lexer.clone()
    code_tokens = []
    lexer.input("<?php " + line)
    while True:
        tok = lexer.token()
        if not tok:
            break
        if tok in ignore:
            continue
        tok = sub_tokens.sub_token(tok,None,allvars)
        code_tokens.append(tok)
    return map_tokens.tokenise(code_tokens)

def getLines(filepath):
    lines = []
    with open(filepath) as f:
        s = f.read()
        inserts_deletes = s.split("@@")
        for phrase in inserts_deletes:
            if phrase is None or phrase == '' or phrase == ' ' or phrase == '\n':
                continue
            out = re.split(" |,|\-",phrase)
            out = list(filter(lambda i: i != '',out))
            if ('+' in out[len(out) - 1]) or ('+' not in out[len(out) - 1] and int(out[len(out) - 1]) > 0):
                lines.append(int(out[0]))
    return lines

def get_tokens(data):
    lexer2 = phplex.lexer.clone()
    lexer2.input(data)
    codetokens = []
    synerror = False
    while True:
        try:
            tok = lexer2.token()
        except IndexError:
            break
        except SyntaxError:
            #print("syntax error :(")
            synerror = True
            break
        if not tok:
            break
        if tok in ignore:
            continue
        tok = sub_tokens.sub_token(tok)
        codetokens.append(tok)
    if synerror:
        return None
    return codetokens

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def _compute_interval(node):
    min_lineno = node.lineno
    max_lineno = node.lineno

    allsubnodes = []
    def visitor_subnodes(node):
        if isinstance(node, ast.Node):
            allsubnodes.append(node)
    node.accept(visitor_subnodes)
    for node in allsubnodes:
        if hasattr(node, "lineno") and node.lineno != None and node.lineno != 0:
            min_lineno = min(min_lineno, node.lineno)
            max_lineno = max(max_lineno, node.lineno)
    return (min_lineno, max_lineno + 1)

def file_to_tree(filename):
    parser = make_parser()
    lexer = phplex.lexer.clone()
    try:
        with open(filename,encoding='utf-8') as f:
            file_content = f.read()
            if len(file_content.split("<?php")) > 2:
                return None
            parsed = parser.parse(file_content, lexer=lexer)
    except UnicodeDecodeError:
        #print("error decoding " + filename)
        return None
    except SyntaxError:
        #print("error syntax " + filename)
        return None
    tree = intervaltree.IntervalTree()
    allnodes = []
    def visitor(node):
        if isinstance(node, ast.Function):
            allnodes.append(node)
        elif isinstance(node, ast.Method):
            allnodes.append(node)

    for node in parsed:
        node.accept(visitor)

    for node in allnodes:
        if isinstance(node, (ast.Function, ast.Class, ast.Method)):
            start, end = _compute_interval(node)
            tree[start:end] = node
    return tree

def get_all_funcs(nodes):
    allfuncs = []
    for node in nodes:
        allfuncs.extend(node.stmt_funcs)
    return allfuncs

vuln_dict = {1:"SQLi",2:"XSS",3:"CI"}

#All logic for running model on input directories is aggregated into function.
#Accepts array of target paths, returns array of results, which are arrays of output strings as in original version.
def run_model(targets):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Recover the model
    model = PhpNetGraphTokensCombine()
    #crutch
    try:
        model.load_state_dict(torch.load("model_combine.pt",device))
    except:
        model.load_state_dict(torch.load(os.path.join(sys.path[0], "model_combine.pt"),device))
    model.to(device)
    model.eval()

    # Perform the scan
    results = []
    for target in targets:
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        output = []

        vulns = 0
        funcs = 0
        predicted = []

        for root, dirs, files in os.walk(target):
            for filename in files:
                file = root + os.sep + filename
                if ".php" in file:
                    matches = file_to_tree(file)
                    # print(matches)
                    if matches is None or len(matches) == 0:
                        matches = [None]

                    if matches:
                        for match in matches:
                            funcs += 1
                            with open(file, "r") as myfile:
                                code_tokens = []
                                lexer = phplex.lexer.clone()
                                parser = make_parser()
                                if match is not None:
                                    interval = min([match], key=lambda i: i[1] - i[0])

                                    func_lines = ["<?php "]
                                    startsopen = False
                                    no_braces = 0
                                    try:
                                        for i, line in enumerate(myfile):
                                            if i >= interval[1]:
                                                break
                                            elif i >= interval[0]:
                                                if line.strip() == "{" and i == interval[0]:
                                                    startsopen = True
                                                    continue
                                                elif line.strip() == "}" and i == (interval[1] - 1) and no_braces == 0:
                                                    continue

                                                if '{' in line:
                                                    no_braces += line.count('{')
                                                if '}' in line:
                                                    no_braces -= line.count('}')
                                                func_lines.append(line)
                                    except UnicodeDecodeError:
                                        #output.append("unidecode error")
                                        continue
                                    data = ''.join(func_lines)
                                else:
                                    try:
                                        data = myfile.read()
                                    except UnicodeDecodeError:
                                        #output.append("unidecode error")
                                        continue
                                try:
                                    nodes = parser.parse(data, lexer=lexer, tracking=True, debug=False)
                                except:
                                    #output.append("parsing error")
                                    #output.append("here " + file)
                                    continue
                                listener = MyPHPListener(name=file)

                                php_traverser.traverse(nodes, listener)

                                cfg = listener.get_graph()
                                allvars = set()
                                for node in list(cfg.nodes):
                                    for var in node.stmt_vars:
                                        allvars.add(var)

                                allfuncs = get_all_funcs(cfg.nodes)
                                edges = [[list(cfg.nodes).index(i), list(cfg.nodes).index(j)] for (i, j) in cfg.edges]


                                if len(edges) == 0:
                                    continue
                                edges = torch.tensor(edges, dtype=torch.long)
                                try:
                                    graph_nodes = torch.tensor([process(node.text, allfuncs, list(allvars)) for node in list(cfg.nodes)])
                                except Exception as e:
                                    #output.append("Tokenising error " + file)
                                    continue
                                data_graph = Data(x=graph_nodes, edge_index=edges.t().contiguous())
                                lexer = phplex.lexer.clone()
                                lexer.input(data)
                                synerror = False
                                while True:
                                    try:
                                        tok = lexer.token()
                                    except IndexError:
                                        break
                                    except SyntaxError:
                                        #output.append("syntax error :(")
                                        synerror = True
                                        break
                                    if not tok:
                                        break
                                    if tok in ignore:
                                        continue
                                    tok = sub_tokens.sub_token(tok,allfuncs,list(allvars))
                                    code_tokens.append(tok)
                                data_tokens = main.get_data_custom_no_y([code_tokens])
                                data_token_in = data_tokens.to(device=device, dtype=torch.long)
                                data_graph_batch = Batch.from_data_list([data_graph]).to(device)
                                pred = model(data_graph_batch,data_token_in)
                                vals = pred.cpu().detach().numpy()
                                preds = np.argmax(vals, axis=1)
                                if preds > 0:
                                    output.append("Found Vuln")
                                    output.append(str(vuln_dict[preds[0]]))
                                    predicted.append(str(vuln_dict[preds[0]]))
                                    output.append("in")
                                    output.append(str(file))
                                    if match is not None:
                                        output.append("function")
                                        output.append(str(match[2]))
                                    #print("code body is")
                                    #print(data)
                                    output.append("NEXT")
                                    vulns+=1
                                else:
                                    predicted.append('Safe')


        output.append('')
        output.append('Total predicted as vulnerable: ' + str(vulns))
        output.append('Total functions: ' + str(funcs))
        output.append('')
        output.append('Predicted distribution:')
        output.append(str(pd.value_counts(predicted)))

        results.append(output)    

    return results

if __name__ == '__main__':
    targets = [path for path in sys.argv[1:] if os.path.isdir(path)]
    if targets:
        results = run_model(targets)
        for target, res in zip(targets, results):
            print("Target", target)
            for line in res:
                print(line)
            print()
    else:
        print("No valid targets provided")