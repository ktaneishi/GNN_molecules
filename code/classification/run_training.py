# %%
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timeit
import sys


# %%
class GraphNeuralNetwork(nn.Module):
    def __init__(self, dim, n_fingerprint, hidden_layer, output_layer, update_func, output_func):
        super(GraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(hidden_layer)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim) for _ in range(output_layer)])
        self.W_property = nn.Linear(dim, 2)
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
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
        '''Update the node vectors in a graph
        considering their neighboring node vectors (i.e., sum or mean),
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

        '''GNN updates the fingerprint vectors.'''
        for i in range(self.hidden_layer):
            fingerprint_vectors = self.update(fingerprint_vectors, adjacencies, M, i)

        if self.output_func == 'sum':
            molecular_vectors = self.sum_axis(fingerprint_vectors, axis)
        if self.output_func == 'mean':
            molecular_vectors = self.mean_axis(fingerprint_vectors, axis)

        for j in range(self.output_layer):
            molecular_vectors = torch.relu(self.W_output[j](molecular_vectors))

        molecular_properties = self.W_property(molecular_vectors)

        return Smiles, molecular_properties

    def __call__(self, data_batch, device, train=True):

        inputs = data_batch[:-1]
        correct_properties = torch.cat(data_batch[-1])
        Smiles, predicted_properties = self.forward(inputs, device)

        if train:
            loss = F.cross_entropy(predicted_properties, correct_properties)
            return loss
        else:
            correct_labels = correct_properties.to('cpu').data.numpy()
            ys = F.softmax(predicted_properties, 1).to('cpu').data.numpy()
            predicted_labels = [np.argmax(y) for y in ys]
            predicted_scores = list(map(lambda x: x[1], ys))
            return Smiles, correct_labels, predicted_labels, predicted_scores


# %%
class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch = batch

    def train(self, dataset, device):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, self.batch):
            data_batch = list(zip(*dataset[i:i+self.batch]))
            loss = self.model(data_batch, device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


# %%
class Tester(object):
    def __init__(self, model, batch):
        self.model = model
        self.batch = batch

    def test(self, dataset, device):
        N = len(dataset)
        SMILES, Ts, Ys, Ss = '', [], [], []

        for i in range(0, N, self.batch):
            data_batch = list(zip(*dataset[i:i+self.batch]))

            Smiles, correct_labels, predicted_labels, predicted_scores = \
                self.model(data_batch, device, train=False)

            SMILES += ' '.join(Smiles) + ' '
            Ts.append(correct_labels)
            Ys.append(predicted_labels)
            Ss.append(predicted_scores)

        SMILES = SMILES.strip().split()
        T = np.concatenate(Ts)
        Y = np.concatenate(Ys)
        S = np.concatenate(Ss)

        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)

        T, Y, S = map(str, T), map(str, Y), map(str, S)
        predictions = '\n'.join(['\t'.join(p) for p in zip(SMILES, T, Y, S)])

        return AUC, precision, recall, predictions

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write(AUCs + '\n')

    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\tScore\n')
            f.write(predictions + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


# %%
def load_tensor(filename, dtype, device):
    return [dtype(d).to(device) for d in np.load(filename + '.npy', allow_pickle=True)]


# %%
def load_numpy(filename):
    return np.load(filename + '.npy', allow_pickle=True)


# %%
def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


# %%
def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


# %%
def main():
    '''Hyperparameters.'''
    DATASET = 'HIV'
    # DATASET = yourdata

    # radius = 1
    radius = 2
    # radius = 3

    update_func = 'sum'
    # update_func = 'mean'

    # output_func = 'sum'
    output_func = 'mean'

    dim = 25
    hidden_layer = 6
    output_layer = 3
    batch = 32
    lr = 1e-3
    lr_decay = 0.9
    decay_interval = 10
    weight_decay = 1e-6
    
    iteration = 300
    iteration = 10
    
    setting = 'default'

    '''CPU or GPU.'''
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU...')

    '''Load preprocessed data.'''
    dir_input = '../../dataset/classification/%s/input/radius%d/' % (DATASET, radius)
    with open(dir_input + 'Smiles.txt') as f:
        Smiles = f.read().strip().split()
    molecules = load_tensor(dir_input + 'molecules', torch.LongTensor, device)
    adjacencies = load_numpy(dir_input + 'adjacencies')
    properties = load_tensor(dir_input + 'properties', torch.LongTensor, device)
    with open(dir_input + 'fingerprint_dict.pkl', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    n_fingerprint = len(fingerprint_dict)

    '''Create a dataset and split it into train/dev/test.'''
    dataset = list(zip(Smiles, molecules, adjacencies, properties))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_test = split_dataset(dataset, 0.8)
    #dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    '''Set a model.'''
    torch.manual_seed(1234)
    model = GraphNeuralNetwork(dim, n_fingerprint, hidden_layer, output_layer, update_func, output_func).to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model, batch)

    '''Output files.'''
    file_AUCs = '../../output/result/AUCs--%s.txt' % setting
    file_predictions = '../../output/result/predictions--%s.txt' % setting
    file_model = '../../output/model/%s.pth' % setting
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_test\tPrecision_test\tRecall_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    '''Start training.'''
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, device)
        #AUC_dev, precision_dev, recall_dev, predictions_dev = tester.test(dataset_dev, device)
        AUC_test, precision_test, recall_test, predictions_test = tester.test(dataset_test, device)

        time = timeit.default_timer() - start

        AUCs = '\t'.join(map(str, [epoch, time, loss_train, #AUC_dev, 
                                   AUC_test, precision_test, recall_test]))
        tester.save_AUCs(AUCs, file_AUCs)
        tester.save_predictions(predictions_test, file_predictions)
        tester.save_model(model, file_model)

        print(AUCs)


# %%
if __name__ == '__main__':
    main()

# %%
