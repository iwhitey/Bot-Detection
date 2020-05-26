import torch
import numpy as np
import IPython

class Glove():

    embedding_dim = 49

    def __init__(self, path, mode='embeddings'):
        self.path = path
        self.mode = mode
        self._read_glove_embeddings()

    def _read_glove_embeddings(self):
        self.glove_dict = {}
        self.glove_dict['<UNK>'] = torch.normal(0, 1, (Glove.embedding_dim,))
        with open(self.path, 'r') as fp:
            #IPython.embed()
            lines = fp.read().split('\n')
            lines = lines[:-1]
        for line in lines:
            line_elements = line.split(' ')
            account = line_elements.pop(0)
            line_elements = torch.as_tensor([float(element) for element in line_elements[1:]])
            #line_elements[1:] = torch.as_tensor(line_elements[1:])
            self.glove_dict[account] = line_elements
            #IPython.embed()
            #self.glove_dict[line_elements[0]] = torch.Tensor(line_elements[1:])
            #print(type(self.glove_dict[line_elements[0]]))
    
    def __len__(self):
        return len(self.glove_dict)

# def avg_embeddings(embedding, data):
#     data_embeddings = torch.zeros(data.size()[0], 300)
#     for i, row in enumerate(data):
#         sum = 0 
#         for idx in row: 
#             sum += embedding(idx) 
#         sum /= row.size()[0]
#         #print(data[i]) 
#         data_embeddings[i] = sum 
#         #print(data_embeddings[i])
#     return data_embeddings

# def embeddings(embedding, data):
#     data_embeddings = torch.zeros(data.size()[0], data.size()[1], 300)
#     for i, row in enumerate(data):
#         for j, idx in enumerate(row):
#             data_embeddings[i, j] = embedding(idx)
#     return data_embeddings



       

