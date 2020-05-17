import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, args):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, args.dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(args.dim, args.dim)] * args.layer_hidden)
        self.W_output = nn.ModuleList([nn.Linear(args.dim, args.dim)] * args.layer_output)
        self.task = args.task

        if self.task == 'classification':
            self.W_property = nn.Linear(args.dim, 2)

        if self.task == 'regression':
            self.W_property = nn.Linear(args.dim, 1)

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
            molecular_vectors = self.gnn(inputs)
            predicted_scores = self.mlp(molecular_vectors)
            loss = F.cross_entropy(predicted_scores, correct_labels)
            return loss
        else:
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_scores = self.mlp(molecular_vectors)
            predicted_scores = predicted_scores.cpu()
            predicted_scores = [s[1] for s in predicted_scores]
            correct_labels = correct_labels.cpu()
            return predicted_scores, correct_labels

    def forward_regressor(self, data_batch, train):
        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            loss = F.mse_loss(predicted_values, correct_values)
            return loss

        else:
            with torch.no_grad():
                molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.cpu()
            correct_values = correct_values.cpu()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return predicted_values, correct_values
