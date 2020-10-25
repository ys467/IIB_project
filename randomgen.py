#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:57:32 2020

@author: yungyu
"""

import pandas as pd
import random
from csv import writer

def append(file_name, list_of_elements):
    #file opened in append mode
    with open(file_name, 'a+', newline='') as write_object:
        #writer object initiated from csv python module
        csv_writer = writer(write_object)
        #contents of list appended as the last row in the csv file
        csv_writer.writerow(list_of_elements)

df = pd.read_csv(
        filepath_or_buffer = 'humidity.csv',
        names = ['Max S', 'Response Time', 'Area Ratio', 'Target'],
        sep = ',')
print(df)

values = df.iloc[:, 0:4].values
i=0

append('humidity.csv',[])

while i<len(df):
    maxs = values[i][0]
    rspt = values[i][1]
    arat = values[i][2]
    trgt = values[i][3]
    j=0
    while j < 20:
        new_row = [random.normalvariate(maxs,2.2),
                   random.normalvariate(rspt,2.2),
                   random.normalvariate(arat,2.2),
                   int(trgt)]
        append('humidity.csv',new_row)
        j += 1
    i+=1
