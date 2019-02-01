import json
from pprint import pprint
import random
import numpy as np
import pandas as pd
import csv
import sys
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import Paths
from Print import Print #pylint: disable=E0401
print = Print()

np.random.seed(1337)

# Clean the data
# List all files types
# find . -type f | sed 's/.*\.//' | sort | uniq -c
# delete everything but json and php
# find temp4 -type f ! -name '*.php' ! -name '*.json'  -print0 | xargs -0 rm -vf

df = pd.read_csv(
        os.path.join(Paths.processed, 'migrations_metadata.csv')
)

print(df.shape)

plt.bar(["files"], df.size, align='center', alpha=0.5)
plt.ylabel('Usage')
plt.title('Data type')
 
plt.show()


print(df.head())

print(1337)
sys.exit()
# See how high a random algoritm work for estimating the datatype

# 1: get all unique values from the data
data = []
with open(r"/Users/anders/Code/github-scrape-laravel/data/processed/migration-analysis-data.csv", 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    headers = next(reader, None)
    for row in reader:
        my_obj = {}
        for index, col in enumerate(row):
            my_obj[headers[index]] = col
        data.append(my_obj)

np.random.shuffle(data)
split_point = int(np.floor(len(data)*0.9))
train = data[0:split_point]
test = data[split_point:-1]

train_datatypes = list(
    map(lambda item: item['datatype'], train)
)
train_unique_datatypes, train_unique_datatypes_count = np.unique(train_datatypes, return_counts=True)

print.info("Lets look at the statistics of the training dataset")
sort_index = np.argsort(-train_unique_datatypes_count)

y_pos = np.arange(len(train_unique_datatypes))

plt.bar(y_pos, train_unique_datatypes_count[sort_index], align='center', alpha=0.5)
plt.xticks(y_pos, train_unique_datatypes[sort_index], rotation=90)
plt.subplots_adjust(bottom=0.35)
plt.ylabel('Usage')
plt.title('Data type')
 
plt.show()


# How many matches if we select a random datatype?
correctGuessRandom = list(
    filter(lambda item: item['datatype'] == np.random.choice(train_unique_datatypes), test)
)
print.info("Random datatype")
print.success("{0:.0%}".format(len(correctGuessRandom)/len(test)), "matches")

# How many matches if we just assume string?
correctGuessString = list(
    filter(lambda item: item['datatype'] == "string", test)
)
print.info("Assuming string")
print.success("{0:.0%}".format(len(correctGuessString)/len(test)), "matches")