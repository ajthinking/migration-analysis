from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import sys
from torch.autograd import Variable
import json
import csv
import numpy as np
import torch.utils.data as data
import time
import operator

from Print import Print
print = Print()

from Network import Network
from MigrationsDataset import MigrationsDataset

migration_train_dataset = MigrationsDataset(train=True, limit_rows=10000)
migration_test_dataset = MigrationsDataset(test=True, limit_rows=10000)

train_loader = DataLoader(
    dataset=migration_train_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=2)

test_loader = DataLoader(
    dataset=migration_test_dataset,
    shuffle=False,
    num_workers=2)    

network = Network(
    number_of_inputs = migration_train_dataset.x.size()[1],
    number_of_outputs = migration_train_dataset.y.size()[1]
)

# Mean Square Error Loss
criterion = nn.MSELoss()

# Stochasctic gradient descent
optimizer = optim.SGD(network.parameters(), lr=1.00)

# training loop
for epoch in range(1000):
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, output = data
        # wrap them in Variable
        inputs, output = Variable(inputs), Variable(output)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = network(inputs)
        # Compute and print loss
        loss = criterion(y_pred, output)
        print("Training epoch", epoch, "loss =", loss)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("\nTesting ------------------------------\n")
successes = 0
fails = 0

first_output = None
for i, data in enumerate(test_loader):
    inputs, output = data
    
    inputs = inputs
    inputs_text = test_loader.dataset.bow_column_name.tensor_to_text(inputs)
    
    prediction = network(inputs).data 
    prediction_text = test_loader.dataset.bow_column_data_type.tensor_to_text(prediction)
    
    actual = output
    actual_text = test_loader.dataset.bow_column_data_type.tensor_to_text(actual)

    if(prediction_text == actual_text):
        print.success(
            inputs_text,
            "successfully interpeted as",
            prediction_text
        )
        successes += 1
    else:
        print.fail(
            inputs_text,
            "interpeted as",
            prediction_text,
            "SHOULD BE",
            actual_text,

        )        
        fails += 1

print("Summary successes", successes, "/", successes + fails)
