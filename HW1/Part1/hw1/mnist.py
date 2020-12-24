"""Problem 3 - Training on MNIST"""
import numpy as np

# TODO: Import any mytorch packages you need (XELoss, SGD, etc)
from mytorch.nn.activations import *
from mytorch.nn.linear import Linear
from mytorch.nn.loss import *
from mytorch.nn.sequential import Sequential
from mytorch.optim.sgd import SGD
from mytorch.tensor import Tensor

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100


def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    model = Sequential(Linear(784, 20), ReLU(), Linear(20, 10))
    optimizer = SGD(model.parameters(), lr=0.1)
    criterion = CrossEntropyLoss()

    # TODO: Call training routine (make sure to write it below)
    val_accuracies = train(model, optimizer, criterion, train_x, train_y, val_x, val_y)
    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    val_accuracies = []
    for epoch in range(num_epochs):
        model.is_train = True
        randomize = np.arange(len(train_x))
        np.random.shuffle(randomize)
        train_x = train_x[randomize]
        train_y = train_y[randomize]
        batches = [(x, y) for x, y in zip(np.vsplit(train_x, train_x.shape[0] / BATCH_SIZE),
                                          np.split(train_y, train_x.shape[0] / BATCH_SIZE))]
        for i, (batch_data, batch_labels) in enumerate(batches):
            optimizer.zero_grad()
            out = model(Tensor(batch_data, requires_grad=False))
            loss = criterion(out, Tensor(batch_labels, requires_grad=False))
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
                model.is_train = True
    return val_accuracies


def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    model.is_train = False
    batches = [(x, y) for x, y in
               zip(np.vsplit(val_x, val_x.shape[0] / BATCH_SIZE), np.split(val_y, val_x.shape[0] / BATCH_SIZE))]
    num_correct = 0
    for (batch_data, batch_labels) in batches:
        out = model(Tensor(batch_data, requires_grad=False))
        batch_preds = np.argmax(out.data, axis=1)
        num_correct += len([batch_preds[i] for i in range(len(batch_preds)) if batch_preds[i] == batch_labels[i]])
    accuracy = num_correct / len(val_x)
    return accuracy
