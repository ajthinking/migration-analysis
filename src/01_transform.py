import sys
import os
from os import walk
from glob import glob
import pandas as pd
import numpy as np
import re
import json

from MigrationFile import MigrationFile
from Print import Print
import Paths

print = Print()

def files():
    files = [y for x in os.walk(Paths.raw) for y in glob(os.path.join(x[0], '*/*/**/*.php'))]
    migration_files = list(map(lambda r: MigrationFile(os.path.realpath(r)), files)) 
    return list(filter(lambda mf: mf.qualifies(), migration_files))

def transform(migration_files):
    rows = []
    for index, migration_file in enumerate(migration_files):
        #print.info("Processing file", index)
        for column_data_type, column_name in migration_file.column_definitions:
            rows.append((
                    migration_file.user,
                    migration_file.repo,
                    migration_file.name,
                    migration_file.table,
                    column_name,
                    column_data_type,
            ))
    return rows

rows = transform(files())

#csv full
df = pd.DataFrame(rows, columns=['user','repo','filename','table','column_name','column_data_type'])
df.index.name='id'
df.to_csv('data/processed/migrations_metadata.csv')

print("Done!")