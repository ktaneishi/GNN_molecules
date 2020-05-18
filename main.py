import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import argparse
import timeit

from model import MolecularGraphNeuralNetwork

def train(dataset, model, optimizer, loss_function, batch_train, epoch):
    train_loss = 0
    model.train()

    for batch_index, index in enumerate(range(0, len(dataset), batch_train), 1):
        data_batch = list(zip(*dataset[index:index+batch_train]))
        correct = torch.cat(data_batch[-1])

        optimizer.zero_grad()
        predicted = model.forward(data_batch)
        loss = loss_function(predicted, correct)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        print('\repoch %4d batch %4d/%4d train_loss %5.3f' % \
                (epoch, batch_index, np.ceil(len(dataset) / batch_train), train_loss / batch_index), end='')

def test(dataset, model, loss_function, batch_test):
    y_score, y_true = [], []
    test_loss = 0
    model.eval()

    for batch_index, index in enumerate(range(0, len(dataset), batch_test), 1):
        data_batch = list(zip(*dataset[index:index+batch_test]))
        correct = torch.cat(data_batch[-1])
        with torch.no_grad():
            predicted = model.forward(data_batch)
        loss = loss_function(predicted, correct)
        test_loss += loss.item()

        y_score.append(predicted.cpu())
        y_true.append(correct.cpu())

    print(' test_loss %5.3f' % (test_loss / batch_index), end='')

    y_score = np.concatenate(y_score)
    y_true = np.concatenate(y_true)

    return y_score, y_true, test_loss / batch_index

def main():
    parser = argparse.ArgumentParser()
    # classification target is a binary value (e.g., drug or not).
    # regression target is a real value (e.g., energy eV).
    parser.add_argument('--task', default='classification', choices=['classification', 'regression'])
    parser.add_argument('--dataset', default='hiv', choices=['hiv', 'photovoltaic'])
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

    filename = 'dataset/%s.npz' % args.dataset
    dataset_train, dataset_test, N_fingerprints = np.load(filename, allow_pickle=True).values()

    for dataset_ in [dataset_train, dataset_test]:
        for index, (fingerprints, adjacency, molecular_size, property) in enumerate(dataset_):
            '''Transform numpy data to torch tensor'''
            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)

            if args.task == 'classification':
                property = torch.LongTensor([int(property)]).to(device)

            if args.task == 'regression':
                property = torch.FloatTensor([[float(property)]]).to(device)

            dataset_[index] = (fingerprints, adjacency, molecular_size, property)

    np.random.shuffle(dataset_train)

    print('# of training data samples:', len(dataset_train))
    print('# of test data samples:', len(dataset_test))

    n_output = 1 if args.task == 'regression' else 2
    model = MolecularGraphNeuralNetwork(N_fingerprints, 
            dim=args.dim, layer_hidden=args.layer_hidden, layer_output=args.layer_output, n_output=n_output).to(device)
    print('# of model parameters:', sum([np.prod(p.size()) for p in model.parameters()]))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.task == 'classification':
        loss_function = F.cross_entropy
    if args.task == 'regression':
        loss_function = F.mse_loss

    test_losses = [np.inf]

    for epoch in range(1, args.epochs+1):
        epoch_start = timeit.default_timer()

        if epoch % args.decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        train(dataset_train, model, optimizer, loss_function, args.batch_train, epoch)
        y_score, y_true, test_loss = test(dataset_test, model, loss_function, args.batch_test)

        if args.task == 'classification':
            y_pred = [np.argmax(x) for x in y_score]
            if len(np.unique(y_pred)) == 2:
                acc = accuracy_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_score[:,1])
                prec = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                print(' test_acc %5.3f test_auc %5.3f test_prec %5.3f test_recall %5.3f' % (acc, auc, prec, recall), end='')

        if args.task == 'regression':
            MAE = np.mean(np.abs(y_score - y_true))
            # mean absolute error
            print(' test_MAE %5.3f' % (MAE), end='')

        print(' %5.1f sec' % (timeit.default_timer() - epoch_start))

        test_losses.append(test_loss)

        if test_loss < min(test_losses[:-1]):
            torch.save(model.state_dict(), 'model.pth')

        if min(test_losses) < min(test_losses[-10:]):
            break

if __name__ == '__main__':
    main()
