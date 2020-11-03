#!/usr/bin/env python

import pandas as pd

col_names = []

colNames = ["Better A", "Better B", "Better C"]
df = pd.read_csv('Sample.csv', names=colNames, index_col=0, header=0)
print(df)



Year
DoY
sod
Navigation mode
Direction mode
Total satellites
Total GNSSs
