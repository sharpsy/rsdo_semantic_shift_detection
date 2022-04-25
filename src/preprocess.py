import argparse
import os
import re
from collections import defaultdict

import dill
import nltk
import pandas as pd
from nltk import sent_tokenize
from nltk.collocations import *
from nltk.corpus import stopwords
from transformers import AutoTokenizer


URL_RE = (
    "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


class Vocab:
    def __init__(self, w_tokenizer, time_tokens):
        self.docs = defaultdict(list)
        self.lemmatized_docs = defaultdict(list)
        self.chunks = []
        self.meta = defaultdict(list)
        self.w_tokenizer = w_tokenizer
        self.time_tokens = time_tokens

    def add(self, doc, lemmatized_doc, chunk, source):
        if chunk not in self.chunks:
            self.chunks.append(chunk)
        self.docs[chunk].append(doc)
        self.lemmatized_docs[chunk].append(lemmatized_doc)
        self.meta[chunk].append(source)

    def make_vocab(self, vocab_path, lang, min_freq):
        print("making_vocab")
        all_freqs = []
        freqs = defaultdict(int)
        punctuation = "!#%'()*+,.:;=?@[\]^`{|}~"
        sw = get_stopwords(lang)
        for chunk in self.chunks:
            print("chunk: ", chunk)
            chunk_freqs = defaultdict(int)
            count_words = 0
            for doc in self.lemmatized_docs[chunk]:
                for sent in doc.split(" <eos> "):
                    for word in sent.split():
                        is_punct = False
                        for p in punctuation:
                            if p in word:
                                is_punct = True
                                break
                        if not is_punct:
                            is_digit = word.isdigit()
                            if not is_digit:
                                if len(word) > 2 and word.lower() not in sw:
                                    chunk_freqs[word] += 1
                                    freqs[word] += 1
                                    count_words += 1
            all_freqs.append((chunk_freqs, count_words))
        print("All vocab size: ", len(freqs))

        filtered_freqs = []
        for word, freq in freqs.items():
            allow = True
            for chunk_freq, _ in all_freqs:
                if chunk_freq[word] < min_freq:
                    allow = False
                    break
            if allow:
                filtered_freqs.append((word, freq))

        print("Length of filtered vocabulary: ", len(filtered_freqs))
        self.freqs = []
        freqs = sorted(filtered_freqs, key=lambda x: x[1], reverse=True)
        with open(vocab_path, "w", encoding="utf8") as f:
            f.write("word,frequency\n")
            for w, freq in freqs:
                w = self.w_tokenizer.tokenize(w)
                # w = "".join(w).replace('##', '')
                w = "".join(w).replace("▁", " ").strip()
                f.write(w + "," + str(freq) + "\n")
                self.freqs.append((w, freq))


def get_stopwords(lang):
    if lang == "slo":
        with open("/app/resources/stopwords.txt", "r", encoding="utf8") as f:
            return set(line.strip() for line in f)
    elif lang == "en":
        return stopwords.words("english")


def preprocess_doc(text, nlp):
    text = (
        text.replace("~", "")
        .replace("­", "")
        .replace("▲", "")
        .replace("_", "")
        .replace("■", "")
        .replace("*", "")
        .replace("^", "")
        .replace("<", "")
        .replace('"', "")
        .replace("'", "")
        .replace("�", "")
        .replace("/", " ")
        .replace("“", "")
        .replace("”", "")
        .replace('"', "")
        .replace("-", " ")
        .replace("–", " ")
        .replace("—", " ")
        .replace("–", " ")
    )
    text = re.sub(URL_RE, "", text)
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    sents = sent_tokenize(text)
    text_filtered = []
    for sent in sents:
        corrupted = False
        if not sent.isupper():
            sent = sent.split()
            sent = " ".join(sent)
            if len(sent) <= 3:
                corrupted = True
            words = sent.split()
            for w in words:
                if w.isupper() and len(w) > 1:
                    corrupted = True
                    break
            if not corrupted:
                text_filtered.append(sent)

    text_filtered = " ".join(text_filtered)
    if len(text_filtered.split()) > 3:
        doc = nlp(text_filtered)

        clean_doc = []
        lemmatized_doc = []
        for sent in doc.sentences:
            original_sent = []
            lemmatized_sent = []
            for token, word in zip(sent.tokens, sent.words):
                lemma = word.lemma.lower()
                if "’" in token.text:
                    lemma = token.text
                if token.ner != "O":
                    lemmatized_sent.append(lemma + "_<ner>")
                else:
                    lemmatized_sent.append(lemma)
                original_sent.append(token.text)

            original_sent = " ".join(original_sent)
            lemmatized_sent = " ".join(lemmatized_sent)
            # print(original_sent)
            # print(lemmatized_sent)
            # print('---------------------------------------------------')
            clean_doc.append(original_sent)
            lemmatized_doc.append(lemmatized_sent)
        return clean_doc, lemmatized_doc
    return None


def preprocess(input_path, output_path, text_column, nlp):
    df_data = pd.read_csv(input_path, sep="\t", encoding="utf8")
    all_data = []
    print("Num docs in the corpus: ", len(df_data))
    counter = 0
    for idx, row in df_data.iterrows():

        counter += 1
        text = row[text_column]
        output = preprocess_doc(text, nlp)
        if output is not None:
            text, lemmatized_text = output
            meta = tuple(row[x] for x in df_data.columns if x != text_column)
            row_data = meta + (" <eos> ".join(text), " <eos> ".join(lemmatized_text))
            all_data.append(row_data)
            print(
                "Processing text: ",
                counter,
                "Tokenized:",
                " <eos> ".join(text)[:100],
                "Lemmatized:",
                " <eos> ".join(lemmatized_text)[:100],
            )
        else:
            print("Discarded row", counter)
    columns = [x for x in df_data.columns if x != text_column] + [
        "preprocessed_text",
        "lemmatized_text",
    ]
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(output_path, sep="\t", encoding="utf8", index=False)
    print("Data preprocessed, preprocessed data saved to", output_path)
    return df


def filter_artefacts(df):
    print("Filtering corpus artefacts")
    num_filtered = 0
    count = 0
    sent_freqs = defaultdict(int)
    for idx, row in df.iterrows():
        text = row["preprocessed_text"]
        sents = text.split("<eos>")
        for sent in sents:
            sent_freqs[sent] += 1

    for idx, row in df.iterrows():
        count += 1
        text = row["preprocessed_text"]
        lemmas = row["lemmatized_text"]
        sents = text.split("<eos>")
        lemmas = lemmas.split("<eos>")
        filtered_sents = []
        filtered_lemmas = []
        for i, sent in enumerate(sents):
            if sent_freqs[sent] < 20:
                filtered_sents.append(sent)
                filtered_lemmas.append(lemmas[i])
            else:
                num_filtered += 1

        sents = "<eos>".join(filtered_sents)
        lemmas = "<eos>".join(filtered_lemmas)
        df.loc[idx, "lemmatized_text"] = lemmas
        df.loc[idx, "preprocessed_text"] = sents
    print("Num sents filtered: ", num_filtered)
    return df


def get_predefined_collocations():
    all_collocations = []
    with open("/app/resources/all_collocations.txt", "r", encoding="utf8") as f:
        for line in f:
            all_collocations.append(tuple(line.strip().split()))
    return list(set(all_collocations))


def label_multiword_expressions(df, lang):
    print("Labelling multiword expressions")
    all_words = []
    for idx, row in df.iterrows():
        text = row["lemmatized_text"]
        sents = text.split("<eos>")
        for sent in sents:
            words = sent.split()
            all_words.extend(words)
    ignored_words = get_stopwords(lang)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder_bigram = BigramCollocationFinder.from_words(all_words, 2)
    finder_bigram.apply_freq_filter(100)
    finder_bigram.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
    bigrams = list(finder_bigram.nbest(bigram_measures.likelihood_ratio, 200))
    finder_trigram = TrigramCollocationFinder.from_words(all_words, 3)
    finder_trigram.apply_freq_filter(100)
    finder_trigram.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
    trigrams = list(finder_trigram.nbest(trigram_measures.likelihood_ratio, 200))
    if lang == "slo":
        predefined_mw = get_predefined_collocations()
        mw_expressions = list(set(bigrams + trigrams + predefined_mw))
    else:
        mw_expressions = list(set(bigrams + trigrams))
    print("Bigrams: ", bigrams)
    print("Trigrams: ", trigrams)
    if lang == "slo":
        print("Predefined: ", predefined_mw)
    del all_words
    del bigrams
    del trigrams
    del finder_trigram
    del finder_bigram
    counter = 0

    for idx, row in df.iterrows():
        counter += 1
        if counter % 1000 == 0:
            print("Going through text num. ", counter)
        lemmatized_text = row["lemmatized_text"]
        text = row["preprocessed_text"]
        for mw in mw_expressions:
            mw_joined = " ".join(mw)
            if mw_joined in lemmatized_text:
                lemmatized_text = lemmatized_text.split()
                text = text.split()
                if len(mw) == 2:
                    num_inserted = 0
                    indices1 = [i for i, x in enumerate(lemmatized_text) if x == mw[0]]
                    indices2 = set(
                        [i for i, x in enumerate(lemmatized_text) if x == mw[1]]
                    )
                    for i in indices1:
                        if i + 1 in indices2:
                            lemmatized_text.insert(i + 1 + num_inserted, "-")
                            text.insert(i + 1 + num_inserted, "-")
                            num_inserted += 1
                elif len(mw) == 3:
                    num_inserted = 0
                    indices1 = [i for i, x in enumerate(lemmatized_text) if x == mw[0]]
                    indices2 = set(
                        [i for i, x in enumerate(lemmatized_text) if x == mw[1]]
                    )
                    indices3 = set(
                        [i for i, x in enumerate(lemmatized_text) if x == mw[2]]
                    )
                    for i in indices1:
                        if i + 1 in indices2 and i + 2 in indices3:
                            lemmatized_text.insert(i + 1 + num_inserted, "-")
                            text.insert(i + 1 + num_inserted, "-")
                            lemmatized_text.insert(i + 3 + num_inserted, "-")
                            text.insert(i + 3 + num_inserted, "-")
                            num_inserted += 2
                if len(text) != len(lemmatized_text):
                    print(
                        "error, len text != len lemmas: ",
                        len(text),
                        len(lemmatized_text),
                    )
                text = " ".join(text).strip()
                lemmatized_text = " ".join(lemmatized_text).strip()
        df.loc[idx, "lemmatized_text"] = lemmatized_text.replace(" - ", "-")
        df.loc[idx, "preprocessed_text"] = text.replace(" - ", "-")
    return df


def _prepare_si_pipeline():
    import classla

    classla.download("sl")
    nlp = classla.Pipeline(lang="sl", processors="tokenize,ner,pos,lemma")
    return nlp


def _prepare_en_pipeline():
    import stanza

    stanza.download("en")
    nlp = stanza.Pipeline(lang="en", processors="tokenize,ner,pos,lemma")
    return nlp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/data/example_data.tsv",
        type=str,
        help="Path to the tsv file containing the data",
    )
    parser.add_argument(
        "--chunks_column",
        default="date",
        type=str,
        help="Name of the column in the data tsv file that should be used for splitting the corpus into chunk.",
    )
    parser.add_argument(
        "--text_column",
        default="text",
        type=str,
        help="Name of the column in the data tsv file containing text",
    )
    parser.add_argument(
        "--lang",
        default="slo",
        type=str,
        help="Language of the corpus, currently only Slovenian ('slo') and English ('en') are supported",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="Path to the folder that will contain generated output vocab and language_model training files",
    )
    parser.add_argument(
        "--min_freq",
        default=10,
        type=int,
        help="Minimum frequency of the word in a specific chunk to be included in the vocabulary",
    )

    args = parser.parse_args()

    nltk.download("punkt")

    assert args.lang in ("en", "slo")
    if args.lang == "slo":
        nlp = _prepare_si_pipeline()
        w_tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta", use_fast=False)
    else:
        nlp = _prepare_en_pipeline()
        w_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    df_data = preprocess(
        args.data_path,
        args.data_path.split(".")[0] + "_preprocessed.tsv",
        args.text_column,
        nlp,
    )
    # df_data = pd.read_csv("data/example_data_preprocessed.tsv", sep='\t', encoding='utf8')
    time_tokens = list(set(df_data[args.chunks_column].tolist()))
    time_tokens = ["<" + str(chunk) + ">" for chunk in time_tokens]
    w_tokenizer.add_tokens(time_tokens)

    vocab_path = os.path.join(args.output_dir, "vocab_list_of_words.csv")
    lm_train_path = os.path.join(args.output_dir, "train_lm.txt")
    lm_test_path = os.path.join(args.output_dir, "test_lm.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df_data = df_data.sample(frac=1, random_state=123)
    df_data = filter_artefacts(df_data)
    df_data = label_multiword_expressions(df_data, args.lang)
    print("Final dataframe shape: ", df_data.shape)

    vocab = Vocab(w_tokenizer, time_tokens)
    all_data = []
    all_sents = []
    all_sources = []
    source_counts = defaultdict(int)
    print(
        "Meta information",
        list(
            x
            for x in df_data.columns
            if x
            not in [
                args.text_column,
                args.chunks_column,
                "preprocessed_text",
                "lemmatized_text",
            ]
        ),
    )
    for idx, row in df_data.iterrows():
        chunk = str(row[args.chunks_column])
        meta = {
            x: row[x]
            for x in df_data.columns
            if x
            not in [
                args.text_column,
                args.chunks_column,
                "preprocessed_text",
                "lemmatized_text",
            ]
        }
        source_counts[chunk] += 1
        text = row["preprocessed_text"]
        lemmatized_text = row["lemmatized_text"]
        sents = text.split(" <eos> ")
        lemmatized_sents = lemmatized_text.split(" <eos> ")
        for sent, lemmatized_sent in zip(sents, lemmatized_sents):
            sent = "<" + chunk + "> " + sent
            lemmatized_sent = "<" + chunk + "> " + lemmatized_sent
            vocab.add(sent, lemmatized_sent, chunk, meta)
            all_sents.append(sent)
        all_sources.append(chunk)
    print("Sources in vocab: ", list(set(all_sources)))

    lm_split = int(len(all_sents) * 0.9)
    train_sents = all_sents[:lm_split]
    test_sents = all_sents[lm_split:]
    train_sents = "\n".join(train_sents)
    test_sents = "\n".join(test_sents)
    with open(lm_train_path, "w", encoding="utf8") as f:
        f.write(train_sents)
    with open(lm_test_path, "w", encoding="utf8") as f:
        f.write(test_sents)

    vocab.make_vocab(vocab_path, args.lang, args.min_freq)
    with open(os.path.join(args.output_dir, "vocab.pickle"), "wb") as handle:
        dill.dump(vocab, handle)

    print("Done building vocab.")
