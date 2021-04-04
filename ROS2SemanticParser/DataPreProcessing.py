from collections import deque
import string
import json
import spacy
import numpy as np


def frame_parse(nl_input):
    # Below is the automated script for generating frames.
    nlp = spacy.load('en_core_web_sm')
    node_list = []

    doc = nlp(nl_input)
    annotation = []
    for token in doc:
        if token.text.lower() == 'left' or token.text.lower() == 'right' or token.text.lower() == 'look':
            annotation.append(0)
        elif token.text.endswith('ing') or token.text.endswith('ed'):
            if token.text.lower() == 'turning':
                annotation.append(1)
            else:
                annotation.append(0)
        elif token.pos_ == 'VERB' or token.text.lower() == 'walk' or token.text.lower() == 'turn':
            annotation.append(1)
        else:
            annotation.append(0)
    # len_of_doc = len(annotation) + 1
    annotation.append(1)
    verb_indices = np.asarray(np.nonzero(annotation)).squeeze()

    pp = verb_indices.size
    for x in range(verb_indices.size-1):
        a = verb_indices[x]
        b = verb_indices[x+1]
        beans = doc[a:b]
        node_list.append(str(beans))
    return node_list


def json_lint(decoded_output):
    list_len = len(decoded_output)
    for i in range(0, list_len):
        if decoded_output[i] in string.punctuation:
            continue
        else:
            if decoded_output[i-1][-1] == "\"":
                x = decoded_output[i-1]
                decoded_output[i-1] = x[:-1] + " "
                decoded_output[i] = decoded_output[i] + '\"'
            else:
                decoded_output[i] = '"{}"'.format(decoded_output[i])

    linted = ''.join(decoded_output)
    return linted


def assemble_graph(umrf_list):
    num_umrfs = len(umrf_list)
    umrfgraph_jsons = {"graph_name": "", "graph_description": ""}
    json_list = []

    i = 0
    for umrf in umrf_list:
        x = json.loads(umrf)
        if i >= 1:
            parent_name = json_list[i-1]['package_name']
            x['parents'] = {'name': parent_name, 'id': "0"}
        i = i + 1
        json_list.append(x)

    if len(json_list) > 1:
        for i in range(0, len(json_list)):
            if i != num_umrfs-1:
                child_name = json_list[i+1]['package_name']
                json_list[i]['children'] = {'name': child_name, 'id': "0"}
    umrfgraph_jsons['umrf_actions'] = json_list

    contents = json.dumps(umrfgraph_jsons)
    return contents

class FIFOBuffer:
    def __init__(self):
        self.data = []
        self.d = deque(self.data)

    def track_sequence(self, nl_input):
        raise NotImplemented

    def set_data(self, data):
        self.data = data
