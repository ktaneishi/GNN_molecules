import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import timeit
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GraphNeuralNetwork(nn.Module):
    def __init__(self, dim, n_fingerprint, hidden_layer, output_layer, update_func, output_func):
        super(GraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)] * hidden_layer)
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)] * output_layer)
        self.W_property = nn.Linear(dim, 2)
        self.update_func = update_func
        self.output_func = output_func

    def pad(self, matrices, pad_value):
        '''Pad adjacency matrices for batch processing.'''
        sizes = [m.shape[0] for m in matrices]
        M = sum(sizes)
        pad_matrices = pad_value + np.zeros((M, M))
        i = 0
        for j, m in enumerate(matrices):
            j = sizes[j]
            pad_matrices[i:i+j, i:i+j] = m
            i += j
        return torch.FloatTensor(pad_matrices)

    def sum_axis(self, xs, axis):
        y = [torch.sum(x, 0) for x in torch.split(xs, axis)]
        return torch.stack(y)

    def mean_axis(self, xs, axis):
        y = [torch.mean(x, 0) for x in torch.split(xs, axis)]
        return torch.stack(y)

    def update(self, xs, A, M, i):
        '''Update the node vectors in a graph considering their neighboring node vectors (i.e., sum or mean),
        which are non-linear transformed by neural network.'''
        hs = torch.relu(self.W_fingerprint[i](xs))
        if self.update_func == 'sum':
            return xs + torch.matmul(A, hs)
        if self.update_func == 'mean':
            return xs + torch.matmul(A, hs) / (M-1)

    def forward(self, inputs, device):
        Smiles, fingerprints, adjacencies = inputs
        axis = [len(f) for f in fingerprints]

        M = np.concatenate([np.repeat(len(f), len(f)) for f in fingerprints])
        M = torch.unsqueeze(torch.FloatTensor(M), 1)

        fingerprints = torch.cat(fingerprints)
        fingerprint_vectors = self.embed_fingerprint(fingerprints)

        adjacencies = self.pad(adjacencies, 0).to(device)

        # GNN updates the fingerprint vectors.
        for i in range(len(self.W_fingerprint)):
            fingerprint_vectors = self.update(fingerprint_vectors, adjacencies, M, i)

        if self.output_func == 'sum':
            molecular_vectors = self.sum_axis(fingerprint_vectors, axis)
        if self.output_func == 'mean':
            molecular_vectors = self.mean_axis(fingerprint_vectors, axis)

        for j in range(len(self.W_output)):
            molecular_vectors = torch.relu(self.W_output[j](molecular_vectors))

        molecular_properties = self.W_property(molecular_vectors)

        return Smiles, molecular_properties

def train(dataset, model, optimizer, batch, device):
    model.train()
    loss_total = 0
    for i in range(0, len(dataset), batch):
        data_batch = list(zip(*dataset[i:i+batch]))       
        inputs = data_batch[:-1]
        correct_properties = torch.cat(data_batch[-1])
        Smiles, predicted_properties = model.forward(inputs, device)
        loss = F.cross_entropy(predicted_properties, correct_properties)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.to('cpu').data.numpy()
    return loss_total

def test(dataset, model, batch, device):
    model.eval()
    SMILES, Ts, Ys, Ss = '', [], [], []

    for i in range(0, len(dataset), batch):
        data_batch = list(zip(*dataset[i:i+batch]))
        inputs = data_batch[:-1]
        correct_properties = torch.cat(data_batch[-1])
        Smiles, predicted_properties = model.forward(inputs, device)

        correct_labels = correct_properties.to('cpu').data.numpy()
        ys = F.softmax(predicted_properties, 1).to('cpu').data.numpy()
        predicted_labels = [np.argmax(y) for y in ys]
        predicted_scores = [x[1] for x in ys]
        
        SMILES += ' '.join(Smiles) + ' '
        Ts.append(correct_labels)
        Ys.append(predicted_labels)
        Ss.append(predicted_scores)

    SMILES = SMILES.strip().split()
    T = np.concatenate(Ts)
    Y = np.concatenate(Ys)
    S = np.concatenate(Ss)

    AUC = roc_auc_score(T, S)
    precision = 0. if np.sum(Y) == 0 else precision_score(T, Y)
    recall = recall_score(T, Y)

    T, Y, S = map(str, T), map(str, Y), map(str, S)
    predictions = '\n'.join(['\t'.join(p) for p in zip(SMILES, T, Y, S)])

    return AUC, precision, recall, predictions

def load_tensor(filename, dtype, device):
    return [dtype(d).to(device) for d in np.load(filename + '.npy', allow_pickle=True)]

def main():
    '''Hyperparameters.'''
    DATASET = 'HIV'
    #DATASET = yourdata

    #radius = 1
    radius = 2
    #radius = 3

    update_func = 'sum'
    #update_func = 'mean'

    #output_func = 'sum'
    output_func = 'mean'

    dim = 25
    hidden_layer = 6
    output_layer = 3
    batch = 32
    lr = 1e-3
    lr_decay = 0.9
    decay_interval = 10
    weight_decay = 1e-6
    
    iteration = 30
    
    setting = 'default'

    # CPU or GPU.
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU...')

    # Load preprocessed data.
    dir_input = '../../dataset/classification/%s/input/radius%d/' % (DATASET, radius)
    with open(dir_input + 'Smiles.txt') as f:
        Smiles = f.read().strip().split()
    molecules = load_tensor(dir_input + 'molecules', torch.LongTensor, device)
    adjacencies = np.load(dir_input + 'adjacencies' + '.npy', allow_pickle=True)
    properties = load_tensor(dir_input + 'properties', torch.LongTensor, device)
    with open(dir_input + 'fingerprint_dict.pkl', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    n_fingerprint = len(fingerprint_dict)

    # Create a dataset and split it into train/test.
    dataset = list(zip(Smiles, molecules, adjacencies, properties))

    np.random.shuffle(dataset)

    dataset_train, dataset_test = train_test_split(dataset, train_size=0.8, test_size=0.2, stratify=properties)
    print('dataset: %d, training: %d, validation: %d' % (len(dataset), len(dataset_train), len(dataset_test)))

    # Set a model.
    model = GraphNeuralNetwork(dim, n_fingerprint, hidden_layer, output_layer, update_func, output_func)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Output files.
    file_predictions = '../../output/result/predictions--%s.txt' % setting
    file_model = '../../output/model/%s.pth' % setting
    columns = ['Epoch', 'Time(sec)', 'Loss_train', 'AUC_val', 'Prec_val', 'Recall_val']

    # Start training.
    print('Training...')
    print(''.join(map(lambda x: '%12s' % x, columns)))
    start = timeit.default_timer()

    for epoch in range(1, iteration):
        if epoch % decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = train(dataset_train, model, optimizer, batch, device)
        AUC_test, precision_test, recall_test, predictions_test = test(dataset_test, model, batch, device)

        time = timeit.default_timer() - start
        start = start + time

        values = [time, loss_train, AUC_test, precision_test, recall_test]
        print('%12s' % epoch + ''.join(map(lambda x: '%12.3f' % x, values)))

    with open(file_predictions, 'w') as out:
        out.write('\t'.join(['Smiles', 'Correct', 'Predict', 'Score']) + '\n')
        out.write(predictions_test + '\n')

    torch.save(model.state_dict(), file_model)

if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    main()
