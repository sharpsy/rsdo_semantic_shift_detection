import argparse
import os
import pickle
import re
from collections import Counter, defaultdict

import nltk
import nltk.corpus
import pandas as pd
import tqdm
from nltk import sent_tokenize
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from pandarallel import pandarallel
from transformers import AutoTokenizer

from vocab import Vocab


URL_RE = (
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
pandarallel.initialize(use_memory_fs=False, progress_bar=True)


def get_stopwords(lang):
    if lang == "slo":
        with open("/app/resources/stopwords.txt", "r", encoding="utf8") as f:
            return set(line.strip() for line in f)
    elif lang == "en":
        return nltk.corpus.stopwords.words("english")


def _preprocess_cleanup(text):
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
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)
    return text


def _filter_noise_sentences(sents):
    text_filtered = []
    for sent in sents:
        if sent.isupper():
            continue
        words = sent.split()
        sent = " ".join(words)
        if len(sent) <= 3:
            # sentence too small
            continue
        # check uppercased word
        sent_corrupted = any((w.isupper() and len(w) > 1) for w in words)
        if sent_corrupted:
            continue
        text_filtered.append(sent)
    text_filtered = " ".join(text_filtered)
    return text_filtered


def _process_doc(text_filtered, lang, _nlp=[None]):
    if _nlp[0] is None:
        if lang == "slo":
            _nlp[0] = _prepare_si_pipeline()
        else:
            _nlp[0] = _prepare_en_pipeline()
    nlp = _nlp[0]
    clean_doc = []
    lemmatized_doc = []
    doc = nlp(text_filtered)
    for sent in doc.sentences:
        original_sent = [tok.text for tok in sent.tokens]
        lemmatized_sent = []
        for token, word in zip(sent.tokens, sent.words):
            if "’" in token.text:
                lemma = token.text
            else:
                lemma = word.lemma.lower()
            if token.ner != "O":
                lemma += "_<ner>"
            lemmatized_sent.append(lemma)

        original_sent = " ".join(original_sent)
        lemmatized_sent = " ".join(lemmatized_sent)
        # print(original_sent)
        # print(lemmatized_sent)
        # print('---------------------------------------------------')
        clean_doc.append(original_sent)
        lemmatized_doc.append(lemmatized_sent)
    return pd.Series(
        [" <eos> ".join(clean_doc), " <eos> ".join(lemmatized_doc)],
        index=["preprocessed_text", "lemmatized_text"],
    )


def preprocess(input_path, output_path, text_column, lang):
    df_data = pd.read_csv(input_path, sep="\t", encoding="utf8")
    print("Num docs in the corpus: ", len(df_data))

    text = df_data[text_column]
    df_data = df_data.drop(columns=[text_column])

    text = text.parallel_apply(_preprocess_cleanup)
    docs_sents = text.parallel_apply(sent_tokenize)
    docs_sents = docs_sents.parallel_apply(_filter_noise_sentences)
    docs_sents = docs_sents[docs_sents.apply(lambda t: len(t.split())) > 3]

    processed_docs = docs_sents.parallel_apply(_process_doc, lang=lang)
    print(processed_docs.to_string(max_colwidth=100))

    df_processed = pd.concat([df_data, processed_docs], axis=1, join="inner")
    print(f"Discarded {len(df_data) - len(df_processed)} rows")
    df_processed.to_csv(output_path, sep="\t", encoding="utf8", index=False)
    print("Data preprocessed, preprocessed data saved to", output_path)
    return df_processed


def filter_artefacts(df):
    print("Filtering corpus artefacts")
    sent_freqs = Counter()
    sentences = df["preprocessed_text"].apply(str.split("<eos>"))
    joined_sentences = sentences.agg(sum)
    sent_freqs.update(joined_sentences)

    num_filtered = 0
    for idx, row in df.iterrows():
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
    all_collocations = set()
    with open("/app/resources/all_collocations.txt", "r", encoding="utf8") as f:
        for line in f:
            collocation = tuple(line.strip().split())
            all_collocations.add(collocation)
    return list(all_collocations)


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
        print("Predefined: ", predefined_mw)
        mw_expressions = list(set(bigrams + trigrams + predefined_mw))
    else:
        mw_expressions = list(set(bigrams + trigrams))
    print("Bigrams: ", bigrams)
    print("Trigrams: ", trigrams)
    del all_words
    del bigrams
    del trigrams
    del finder_trigram
    del finder_bigram

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        lemmatized_text = row["lemmatized_text"]
        text = row["preprocessed_text"]
        for mw in mw_expressions:
            mw_joined = " ".join(mw)
            if mw_joined not in lemmatized_text:
                continue
            lemmatized_text = lemmatized_text.split()
            text = text.split()
            assert len(mw) in (2, 3)
            num_inserted = 0
            indices1 = [i for i, x in enumerate(lemmatized_text) if x == mw[0]]
            indices2 = set([i for i, x in enumerate(lemmatized_text) if x == mw[1]])
            if len(mw) == 2:
                for i in indices1:
                    if i + 1 in indices2:
                        lemmatized_text.insert(i + 1 + num_inserted, "-")
                        text.insert(i + 1 + num_inserted, "-")
                        num_inserted += 1
            elif len(mw) == 3:
                indices3 = set([i for i, x in enumerate(lemmatized_text) if x == mw[2]])
                for i in indices1:
                    if i + 1 in indices2 and i + 2 in indices3:
                        lemmatized_text.insert(i + 1 + num_inserted, "-")
                        text.insert(i + 1 + num_inserted, "-")
                        lemmatized_text.insert(i + 3 + num_inserted, "-")
                        text.insert(i + 3 + num_inserted, "-")
                        num_inserted += 2
            if len(text) != len(lemmatized_text):
                txt_len, lemm_len = len(text), len(lemmatized_text)
                print(
                    "error, length of text and lemmas is not the same "
                    f"[text:{txt_len}, lemmas:{lemm_len}] "
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
    os.makedirs(args.output_dir, exist_ok=True)

    assert args.lang in ("en", "slo")
    if args.lang == "slo":
        w_tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta", use_fast=False)
    else:
        w_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    df_data = preprocess(
        args.data_path,
        os.path.join(args.output_dir, "preprocessed.tsv"),
        args.text_column,
        args.lang,
    )
    # df_data = pd.read_csv("data/example_data_preprocessed.tsv", sep='\t', encoding='utf8')
    time_tokens = df_data[args.chunks_column].drop_duplicates()
    time_tokens = "<" + time_tokens + ">"
    w_tokenizer.add_tokens(time_tokens.tolist())

    vocab_path = os.path.join(args.output_dir, "vocab_list_of_words.csv")
    lm_train_path = os.path.join(args.output_dir, "train_lm.txt")
    lm_test_path = os.path.join(args.output_dir, "test_lm.txt")

    df_data = df_data.sample(frac=1, random_state=123)
    df_data = filter_artefacts(df_data)
    df_data = label_multiword_expressions(df_data, args.lang)
    print("Final dataframe shape: ", df_data.shape)

    vocab = Vocab(w_tokenizer, time_tokens)
    all_data = []
    all_sents = []
    all_sources = set()
    source_counts = defaultdict(int)
    meta_columns = [
        col
        for col in df_data.columns
        if col
        not in (
            args.text_column,
            args.chunks_column,
            "preprocessed_text",
            "lemmatized_text",
        )
    ]
    print("Meta information:", meta_columns)
    for idx, row in df_data.iterrows():
        chunk = str(row[args.chunks_column])
        meta = {x: row[x] for x in meta_columns}
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
        all_sources.add(chunk)
    print("Sources in vocab: ", list(all_sources))

    lm_split = int(len(all_sents) * 0.9)
    train_sents = all_sents[:lm_split]
    test_sents = all_sents[lm_split:]
    train_sents = "\n".join(train_sents)
    test_sents = "\n".join(test_sents)
    with open(lm_train_path, "w", encoding="utf8") as f:
        f.write(train_sents)
    with open(lm_test_path, "w", encoding="utf8") as f:
        f.write(test_sents)

    stopwords = get_stopwords(args.lang)
    vocab.make_vocab(vocab_path, args.lang, args.min_freq, stopwords)
    with open(os.path.join(args.output_dir, "vocab.pickle"), "wb") as vocab_file:
        pickle.dump(vocab, vocab_file)

    print("Done building vocab.")
