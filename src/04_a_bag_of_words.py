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

from Network import Network
from MigrationsDataset import MigrationsDataset

def tensor_to_string(tensor):
    result = ''
    for i, v in enumerate(tensor[0]):
        result += str(round(1000*v.item())) + " "
    return result

migration_train_dataset = MigrationsDataset(train=True, limit_rows=1000)
migration_test_dataset = MigrationsDataset(test=True, limit_rows=1000)

train_loader = DataLoader(
    dataset=migration_train_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=2)

test_loader = DataLoader(
    dataset=migration_test_dataset,
    shuffle=False,
    num_workers=2)    


print(
    # test_loader.dataset.x[0]
    test_loader.dataset.bow_column_name.tensor_to_text(
        test_loader.dataset.x[0]
    )
)

sys.exit()


network = Network(
    number_of_inputs = migration_train_dataset.x.size()[1],
    number_of_outputs = migration_train_dataset.y.size()[1]
)

#sys.exit()

# Mean Square Error Loss
criterion = nn.MSELoss()
# Stochasctic gradient descent

optimizer = optim.SGD(network.parameters(), lr=1.0)

# training loop
for epoch in range(100):
    print("Training epoch", epoch)
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, output = data
        # wrap them in Variable
        inputs, output = Variable(inputs), Variable(output)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = network(inputs)
        # Compute and print loss
        loss = criterion(y_pred, output)
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
    prediction = network(inputs).data 
    actual = output
    inputs_index, inputs_value = max(enumerate(inputs), key=operator.itemgetter(1))
    prediction_index, prediction_value = max(enumerate(actual), key=operator.itemgetter(1))
    

    # print(
    #     inputs_index,
    #     train_loader.dataset.input_tensor_to_text(inputs),
    #     inputs,
    #     prediction_index,
    #     train_loader.dataset.output_tensor_to_text(actual),
    #     actual
    # )


    # print(
    #         "prediction for",
    #         test_loader.dataset.input_tensor_to_text(inputs),
    #         ":", 
    #         test_loader.dataset.output_tensor_to_text(prediction),
    #         "actual",
    #         test_loader.dataset.output_tensor_to_text(actual)
    # )

    if(test_loader.dataset.bow_column_name.tensor_to_text(prediction) == test_loader.dataset.bow_column_name.tensor_to_text(actual)):
        successes += 1
    else:
        fails += 1
    # print(prediction)
    # print(actual)
    # print("--------")


# print("Summary successes", successes, "/", successes + fails)
