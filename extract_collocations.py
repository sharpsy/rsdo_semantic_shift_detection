import os
import pandas as pd

folder = 'GF2.1-Collocations_JOS-structures'
all_files = os.listdir(folder)
all_files = [x for x in all_files if x.endswith('.csv')]

all = []
for f in all_files:
    df = pd.read_csv(os.path.join(folder,f), encoding='utf8', sep=',')
    for idx, row in df.iterrows():
        w1=  str(row['C1_Lemma'])
        w2 = str(row['C2_Lemma'])
        w3 = str(row['C3_Lemma'])
        if int(row['Frequency']) > 100:
            collocation = w1 + ' ' + w2
            if w3 != 'nan':
                collocation = collocation + ' ' + w3
                all.append((collocation, int(row['Frequency'])))
    all = sorted(all, key=lambda x: x[1], reverse=True)[:600]

with open('data/all_collocations.txt', 'w', encoding='utf8') as o:
    for collocation, freq in all:
        print(collocation, freq)
        o.write(collocation.strip() + '\n')

