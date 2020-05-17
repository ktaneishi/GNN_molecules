import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import argparse
import timeit

from model import MolecularGraphNeuralNetwork

def train(dataset, model, optimizer, batch_train, epoch):
    train_loss = 0
    model.train()

    for index in range(0, len(dataset), batch_train):
        data_batch = list(zip(*dataset[index:index+batch_train]))
        optimizer.zero_grad()

        if model.task == 'classification':
            loss = model.forward_classifier(data_batch, train=True)

        if model.task == 'regression':
            loss = model.forward_regressor(data_batch, train=True)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        print('\repoch %4d train_loss %6.3f' % (epoch, train_loss / index), end='')

def test_classifier(dataset, model, batch_test):
    y_score, y_true = [], []
    model.eval()

    for index in range(0, len(dataset), batch_test):
        data_batch = list(zip(*dataset[index:index+batch_test]))
        with torch.no_grad():
            predicted_scores, correct_labels = model.forward_classifier(data_batch, train=False)
        y_score.append(predicted_scores)
        y_true.append(correct_labels)

    acc = np.equal(np.concatenate(y_score) > 0.5, np.concatenate(y_true)).sum() / len(dataset)

    print(' test_acc %6.3f' % (acc), end='')

def test_regressor(dataset, model, batch_test):
    N = len(dataset)
    SAE = 0 # sum absolute error.
    model.eval()

    for index in range(0, N, batch_test):
        data_batch = list(zip(*dataset[index:index+batch_test]))
        with torch.no_grad():
            predicted_values, correct_values = model.forward_regressor(data_batch, train=False)
        SAE += sum(np.abs(predicted_values-correct_values))

    MAE = SAE / N  # mean absolute error.
    print(' test_MAE %6.3f' % (MAE), end='')

def main():
    parser = argparse.ArgumentParser()
    # classification target is a binary value (e.g., drug or not).
    # regression target is a real value (e.g., energy eV).
    parser.add_argument('--task', default='classification', choices=['classification', 'regression'])
    parser.add_argument('--dataset', default='hiv', choices=['hiv', 'photovoltaic'])
    parser.add_argument('--radius', default=1)
    parser.add_argument('--dim', default=50)
    parser.add_argument('--layer_hidden', default=6)
    parser.add_argument('--layer_output', default=6)
    parser.add_argument('--batch_train', default=32)
    parser.add_argument('--batch_test', default=32)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--lr_decay', default=0.99)
    parser.add_argument('--decay_interval', default=10)
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--random_seed', default=123)

    args = parser.parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    filename = 'dataset/%s-%s.npz' %(args.task, args.dataset)
    dataset_train, dataset_test, N_fingerprints = np.load(filename, allow_pickle=True).values()

    for dataset_ in [dataset_train, dataset_test]:
        for index, (fingerprints, adjacency, molecular_size, prop) in enumerate(dataset_):
            '''Transform numpy data to torch tensor'''
            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)

            if args.task == 'classification':
                prop = torch.LongTensor([int(prop)]).to(device)

            if args.task == 'regression':
                prop = torch.FloatTensor([[float(prop)]]).to(device)

            dataset_[index] = (fingerprints, adjacency, molecular_size, prop)

    np.random.shuffle(dataset_train)

    print('# of training data samples:', len(dataset_train))
    print('# of test data samples:', len(dataset_test))

    print('Creating a model.')
    model = MolecularGraphNeuralNetwork(N_fingerprints, args).to(device)
    print('# of model parameters:', sum([np.prod(p.size()) for p in model.parameters()]))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        epoch_start = timeit.default_timer()

        if epoch % args.decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        train(dataset_train, model, optimizer, args.batch_train, epoch)

        if args.task == 'classification':
            test_classifier(dataset_test, model, args.batch_test)

        if args.task == 'regression':
            test_regressor(dataset_test, model, args.batch_test)

        print(' %5.1f sec' % (timeit.default_timer() - epoch_start))

if __name__ == '__main__':
    main()
