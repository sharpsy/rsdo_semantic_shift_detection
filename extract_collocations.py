import pathlib

import pandas as pd


NUM_COLLOCATIONS = 600
FREQ_THRESHOLD = 100
ROOT_DIR = pathlib.Path("GF2.1-Collocations_JOS-structures")

subframes = []
for fpath in ROOT_DIR.iterdir():
    if not fpath.is_file() or not fpath.suffix == ".csv":
        continue
    subdf = pd.read_csv(fpath, encoding="utf8")
    subdf = df[df["Frequency"] > FREQ_THRESHOLD]
    subdf = df.dropna(subset="C3_Lemma")
    subframes.append(subdf)

df = pd.concat(subframes)
df = df.sort_values(by=["Frequency"], ascending=False)
df = all_df.iloc[:NUM_COLLOCATIONS]
print(f"Top {NUM_COLOCATION} collocations:")
print(df.to_csv(index=False))
collocations = df[["C1_Lemma", "C2_Lemma", "C3_Lemma"]].apply(" ".join)
collocations = collocation.apply(str.strip)
collocation.to_csv("data/all_collocations.txt", header=False, index=False)
