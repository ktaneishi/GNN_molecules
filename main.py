import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import argparse
import timeit

from model import MolecularGraphNeuralNetwork

def data_load(args, device):
    filename = 'dataset/%s.npz' % args.dataset
    dataset_train, dataset_test, N_fingerprints = np.load(filename, allow_pickle=True).values()

    '''Transform numpy data to torch tensor'''
    for index, (fingerprints, adjacency, molecular_size, property) in enumerate(dataset_train):
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        if args.task == 'classification':
            property = torch.LongTensor([int(property)]).to(device)
        if args.task == 'regression':
            property = torch.FloatTensor([[float(property)]]).to(device)
        dataset_train[index] = (fingerprints, adjacency, molecular_size, property)

    for index, (fingerprints, adjacency, molecular_size, property) in enumerate(dataset_test):
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        if args.task == 'classification':
            property = torch.LongTensor([int(property)]).to(device)
        if args.task == 'regression':
            property = torch.FloatTensor([[float(property)]]).to(device)
        dataset_test[index] = (fingerprints, adjacency, molecular_size, property)

    np.random.shuffle(dataset_train)

    return dataset_train, dataset_test, N_fingerprints

def train(dataset, net, optimizer, loss_function, batch_train, epoch):
    train_loss = 0
    net.train()

    for batch_index, index in enumerate(range(0, len(dataset), batch_train), 1):
        data_batch = list(zip(*dataset[index:index+batch_train]))
        correct = torch.cat(data_batch[-1])

        optimizer.zero_grad()
        predicted = net.forward(data_batch)
        loss = loss_function(predicted, correct)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    print('epoch %4d batch %4d/%4d train_loss %5.3f' % \
            (epoch, batch_index, np.ceil(len(dataset) / batch_train), train_loss / batch_index), end='')

def test(dataset, net, loss_function, batch_test):
    test_loss = 0
    net.eval()

    for batch_index, index in enumerate(range(0, len(dataset), batch_test), 1):
        data_batch = list(zip(*dataset[index:index+batch_test]))
        correct = torch.cat(data_batch[-1])
        with torch.no_grad():
            predicted = net.forward(data_batch)
        loss = loss_function(predicted, correct)
        test_loss += loss.item()

    print(' test_loss %5.3f' % (test_loss / batch_index), end='')

    return test_loss / batch_index

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % device)

    dataset_train, dataset_test, N_fingerprints = data_load(args, device)

    print('# of training data samples:', len(dataset_train))
    print('# of test data samples:', len(dataset_test))

    n_output = 1 if args.task == 'regression' else 2
    net = MolecularGraphNeuralNetwork(N_fingerprints, dim=args.dim, 
            layer_hidden=args.layer_hidden, layer_output=args.layer_output, n_output=n_output).to(device)
    print('# of model parameters:', sum([np.prod(p.size()) for p in net.parameters()]))

    if args.modelfile:
        net.load_state_dict(torch.load(args.modelfile))

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    loss_function = F.cross_entropy if args.task == 'classification' else F.mse_loss

    test_losses = []

    for epoch in range(args.epochs):
        epoch_start = timeit.default_timer()

        if epoch % args.decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        train(dataset_train, net, optimizer, loss_function, args.batch_train, epoch)
        test_loss = test(dataset_test, net, loss_function, args.batch_test)

        print(' %5.2f sec' % (timeit.default_timer() - epoch_start))

        test_losses.append(test_loss)

        if len(test_losses) > 1 and test_loss < min(test_losses[:-1]):
            torch.save(net.state_dict(), 'model/%5.3f.pth' % test_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classification target is a binary value (e.g., drug or not).
    # regression target is a real value (e.g., energy eV).
    parser.add_argument('--task', default='classification', choices=['classification', 'regression'])
    parser.add_argument('--dataset', default='hiv', choices=['hiv', 'photovoltaic'])
    parser.add_argument('--modelfile', default=None)
    parser.add_argument('--dim', default=50)
    parser.add_argument('--layer_hidden', default=6)
    parser.add_argument('--layer_output', default=6)
    parser.add_argument('--batch_train', default=32)
    parser.add_argument('--batch_test', default=32)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--lr_decay', default=0.99)
    parser.add_argument('--decay_interval', default=10)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', default=123)
    args = parser.parse_args()
    print(vars(args))

    main(args)
