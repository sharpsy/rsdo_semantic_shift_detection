# Semantic Change Detection

Instructions for use assume availability of Docker and GNU Make.<br/>


## Prepare configuration.

Edit `config.mk` file and update values according to your configuration.

## Run analysis ##

```
make all
```

Generated results are placed in the newly created `out` direcotry.


## Running each step separately ##

If you want to run each analysis step separately, you can run:

1) Build the docker image
```
make build
```

This step will prepare a docker image and install dependencies.


2) Preprocess data
```
make preprocess
```

This step takes in a corpus in the tsv format (see 'data/example_data.csv' for example) as an input and generates preprocessed file to be used in later steps. Additionaly, vocabulary and train/test splits for language modelling  are prepared.

Generated outputs:
- **out/preprocessed.tsv** saved to the folder containing input data
- **out/vocab.pickle** Serialized vocabulary data
- **out/train_lm.txt** Train corpus for language model fine-tuning
- **out/test_lm.txt** Test corpus for language model fine-tuning
- **out/vocab_list_of_words.csv** Words for which semantic shift will be calculated


3) Finetune language model

```
make finetune
```

By default a RoBERTa language model is used.

Generated outputs:
- **out/models/** - fine-tuned LM model that can be used for embedding generation at a later step

4) Extract embeddings

```
make embeddings
```

This will infer vectors specific to each data sub-group.

Generated outputs:
- **out/embeddings.pickle** output file containing embeddings

5) Measure semantic shift between data sub-groups

```
make semantic-shift
```

Generated outputs:
- **out/word_list_results.csv** A list of all words in the vocab with their semantic change scores
- **out/corpus_slices.pkl** Serialized list of corpus slices used at a later stage
- **out/id2sents.pkl** Serialized sentences used at a later stage
- **out/sents.pkl** Serialized sentences used at a later stage
- **out/kmeans_5_labels.pkl** Serialized cluster labels for each word used at a later stage

6) Plot results for target words

```
make interpretation
```

Extract keywords for each word usage cluster and plot clusters distributions

Generated outputs:
- **out/results/images/*.png** An image showing a distribution of word usages for each target word
- **out/results/docs/*.tsv** A tsv document for each target word containing information about sentences in which it appeared and which cluster it belongs to





