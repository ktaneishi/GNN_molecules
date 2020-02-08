#!/usr/bin/env python
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import preprocess as pp
import timeit
import sys
import os

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, task):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)] * layer_hidden)
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)] * layer_output)
        self.task = task
        if self.task == 'classification':
            self.W_property = nn.Linear(dim, 2)
        if self.task == 'regression':
            self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        '''Pad the list of matrices with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C], we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        '''
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(matrices[0].device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        '''Cat or pad each input data for batch processing.'''
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        '''GNN layer (update the fingerprint vectors).'''
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(len(self.W_fingerprint)):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        '''Molecular vector by sum or mean of the fingerprint vectors.'''
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

    def mlp(self, vectors):
        '''Classifier or regressor based on multilayer perceptron.'''
        for l in range(len(self.W_output)):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs

    def forward_classifier(self, data_batch, train):
        inputs = data_batch[:-1]
        correct_labels = torch.cat(data_batch[-1])

        if train:
            self.train()
            molecular_vectors = self.gnn(inputs)
            predicted_scores = self.mlp(molecular_vectors)
            loss = F.cross_entropy(predicted_scores, correct_labels)
            return loss
        else:
            self.eval()
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_scores = self.mlp(molecular_vectors)
            predicted_scores = predicted_scores.to('cpu').detach().numpy()
            predicted_scores = [s[1] for s in predicted_scores]
            correct_labels = correct_labels.to('cpu').detach().numpy()
            return predicted_scores, correct_labels

    def forward_regressor(self, data_batch, train):
        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            self.train()
            molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss
        else:
            self.eval()
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').detach().numpy()
            correct_values = correct_values.to('cpu').detach().numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return predicted_values, correct_values

class Trainer(object):
    def __init__(self, model, lr):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset, batch_train):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            if self.model.task == 'classification':
                loss = self.model.forward_classifier(data_batch, train=True)
            if self.model.task == 'regression':
                loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_classifier(self, dataset, batch_test):
        N = len(dataset)
        y_score, y_true = [], []
        loss_total = 0
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            predicted_scores, correct_labels = self.model.forward_classifier(data_batch, train=False)
            y_score.append(predicted_scores)
            y_true.append(correct_labels)
        acc = np.equal(np.concatenate(y_score) > 0.5, np.concatenate(y_true)).sum()
        return acc

    def test_regressor(self, dataset, batch_test):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            predicted_values, correct_values = self.model.forward_regressor(data_batch, train=False)
            SAE += sum(np.abs(predicted_values-correct_values))
        MAE = SAE / N  # mean absolute error.
        return MAE

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

def main():
    task = 'classification' # target is a binary value (e.g., drug or not).
    dataset = 'hiv'

    #task = 'regression' # target is a real value (e.g., energy eV).
    #dataset = 'photovoltaic'

    radius = 1
    dim = 50
    layer_hidden = 6
    layer_output = 6

    batch_train = 32
    batch_test = 32
    lr = 1e-4
    lr_decay = 0.99
    decay_interval = 10
    iteration = 10 #1000

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    filename = '%s-%s.npz' %(task, dataset)

    setting = '%s-%d-%d-%d-%d-%d-%d-%f-%f-%d-%d' % (
            dataset, radius, dim, layer_hidden, layer_output, batch_train, batch_test,
            lr, lr_decay, decay_interval, iteration)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    if os.path.exists(filename):
        dataset_train, dataset_test, N_fingerprints = np.load(filename, allow_pickle=True).values()
    else:
        (dataset_train, dataset_test, N_fingerprints) = pp.create_datasets(task, dataset, radius)
        np.savez_compressed(filename, dataset_train=dataset_train, dataset_test=dataset_test, N_fingerprints=N_fingerprints)

    for dataset in [dataset_train, dataset_test]:
        for index, (fingerprints, adjacency, molecular_size, property) in enumerate(dataset):
            '''Transform the above each data of numpy'''
            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)
            if task == 'classification':
                property = torch.LongTensor([int(property)]).to(device)
            if task == 'regression':
                property = torch.FloatTensor([[float(property)]]).to(device)
            dataset[index] = (fingerprints, adjacency, molecular_size, property)

    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of test data samples:', len(dataset_test))

    print('Creating a model.')
    model = MolecularGraphNeuralNetwork(N_fingerprints, dim, layer_hidden, layer_output, task).to(device)
    trainer = Trainer(model, lr)
    tester = Tester(model)
    print('# of model parameters:', sum([np.prod(p.size()) for p in model.parameters()]))

    file_result = '../output/result--' + setting + '.txt'
    if task == 'classification':
        result = '%5s%12s%12s%12s' % ('epoch', 'train_loss', 'test_acc', 'time(sec)')
    if task == 'regression':
        result = '%5s%12s%12s%12s' % ('epoch', 'train_loss', 'test_MAE', 'time(sec)')

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    start = timeit.default_timer()

    for epoch in range(1, iteration+1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, batch_train)

        if task == 'classification':
            #prediction_test = tester.test_classifier(dataset_test, batch_test)
            test_acc = tester.test_classifier(dataset_test, batch_test)
        if task == 'regression':
            prediction_test = tester.test_regressor(dataset_test, batch_test)

        time = timeit.default_timer() - start
        start = start + time

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about', hours, 'hours', minutes, 'minutes.')
            print(result)

        result = '%5d%12.4f%12.4f%12.4f' % (epoch, loss_train, test_acc, time)
        tester.save_result(result, file_result)

        print(result)

if __name__ == '__main__':
    main()
