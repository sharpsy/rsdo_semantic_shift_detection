import argparse
import os
from collections import Counter, defaultdict

import dill
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import unidecode
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def get_stopwords(lang):
    if lang == "slo":
        sw = []
        with open("resources/stopwords.txt", "r", encoding="utf8") as f:
            for line in f:
                sw.append(unidecode.unidecode(line.strip()))
        return set(sw)
    elif lang == "en":
        return set(stopwords.words("english"))


def get_clusters_sent(
    target,
    threshold_size_cluster,
    labels,
    sentences,
    id2sents,
    corpus_slices,
    docs_folder,
):

    labels = dill.load(open(labels, "rb"))
    sentences = dill.load(open(sentences, "rb"))
    id2sents = dill.load(open(id2sents, "rb"))

    cluster_to_sentence = defaultdict(lambda: defaultdict(list))
    for cs in corpus_slices:
        for label, sents in zip(labels[target][cs], sentences[target][cs]):
            for sent in sents:
                sent_id = int(str(corpus_slices.index(cs) + 1) + str(sent))
                sent = id2sents[sent_id]
                cluster_to_sentence[label][cs].append(sent)

    counts = {cs: Counter(labels[target][cs]) for cs in corpus_slices}
    all_labels = []
    for slice, c in counts.items():
        slice_labels = [x[0] for x in c.items()]
        all_labels.extend(slice_labels)
    all_labels = set(all_labels)
    all_counts = []
    for l in all_labels:
        all_count = 0
        for slice in corpus_slices:
            count = counts[slice][l]
            all_count += count
        all_counts.append((l, all_count))

    sorted_counts = sorted(all_counts, key=lambda x: x[1], reverse=True)
    sentences = []
    lemmas = []
    metas = []
    labels = []
    categs = []

    for label, count in sorted_counts:
        if count > threshold_size_cluster:
            for cs in corpus_slices:
                for (sent, lemma, meta) in cluster_to_sentence[label][cs]:
                    sent_clean = sent.strip()
                    lemma_clean = lemma.replace("_<ner>", "").strip()
                    time_token = "<" + str(cs) + ">"
                    sent_clean = sent_clean.replace(time_token, "").strip()
                    lemma_clean = lemma_clean.replace(time_token, "").strip()
                    if sent_clean not in set(sentences):
                        sentences.append(sent_clean)
                        lemmas.append(lemma_clean)
                        labels.append(label)
                        categs.append(cs)
                        metas.append(meta)
                        # print(sent_clean)
        else:
            print("Cluster", label, "is too small - deleted!")
    all_metas = []
    if metas:
        columns = list(metas[0].keys())
        meta_values = [list(x.values()) for x in metas]
        for i in range(len(metas[0])):
            meta_value = [x[i] for x in meta_values]
            all_metas.append(meta_value)
    else:
        columns = []
    columns = columns + ["slice", "cluster_label", "sentence", "lemmatized_sent"]

    output = all_metas + [categs, labels, sentences, lemmas]
    sent_df = pd.DataFrame(list(zip(*output)), columns=columns)
    sent_df.to_csv(
        os.path.join(docs_folder, target + "_sentences.tsv"),
        encoding="utf-8",
        index=False,
        sep="\t",
    )
    return sent_df


def output_distrib(data, word, keyword_clusters, image_folder):
    distrib = data.groupby(["slice", "cluster_label"]).size().reset_index(name="count")
    pivot_distrib = distrib.pivot(
        index="slice", columns="cluster_label", values="count"
    )
    pivot_distrib_norm = pivot_distrib.div(pivot_distrib.sum(axis=1), axis=0)
    pivot_distrib_norm = pivot_distrib_norm.fillna(0)
    first_column = pivot_distrib_norm.columns[0]
    order = list("Slice " + x for x in pivot_distrib_norm[first_column].keys())
    columns = []
    final_data = []
    for i in keyword_clusters:
        name = "Cluster " + str(i) + ": " + ", ".join(keyword_clusters[i][:7])
        distrib = np.array(list(pivot_distrib_norm[i].fillna(0).array))
        final_data.append((name, distrib))

    final_data = sorted(final_data, reverse=True, key=lambda x: sum(x[1]))
    if len(final_data) <= 10:
        for name, distrib in final_data:
            columns.append(go.Bar(name=name, x=order, y=distrib))
    else:
        for name, distrib in final_data[:9]:
            columns.append(go.Bar(name=name, x=order, y=distrib))
        other_data = final_data[9:]
        other = None
        print(other_data)
        for name, distrib in other_data:
            if other is None:
                other = distrib
                print(distrib, other)
            else:
                other += distrib
                print(distrib, other)

        print("Other: ", other)
        columns.append(go.Bar(name="Other", x=order, y=other))

    fig = go.Figure(data=columns)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        width=1200,
        height=800,
        barmode="stack",
        title="",
        xaxis_title=f"{word}",
        yaxis_title="Distribution",
        legend_title="",
        font=dict(size=14, color="Black"),
        legend=dict(yanchor="top", y=1.4, xanchor="left", x=0.3),
    )
    # fig.show()
    fig.write_image(os.path.join(image_folder, f"{word}.png"))

    return pivot_distrib_norm


def extract_topn_from_vector(feature_names, sorted_items, topn):
    """get the feature names and tf-idf score of top n items"""
    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results


def extract_keywords(target_word, word_clustered_data, max_df, topn, lang):
    sw = get_stopwords(lang)
    # get groups of sentences for each cluster
    l_sent_clust_dict = defaultdict(list)
    sent_clust_dict = defaultdict(list)
    for i, row in word_clustered_data.iterrows():
        sent = " ".join(row["sentence"].split())
        lemmatized_sent = " ".join(row["lemmatized_sent"].split())
        l_sent_clust_dict[row["cluster_label"]].append((sent, lemmatized_sent))

    for label, data in l_sent_clust_dict.items():
        original_sents = "\t".join([x[0] for x in data])
        lemmas = "\t".join([x[1] for x in data])
        sent_clust_dict[label] = (original_sents, lemmas)

    labels = []
    lemmatized_clusters = []
    for label, (sents, lemmatized_sents) in sent_clust_dict.items():
        labels.append(label)
        lemmatized_clusters.append(lemmatized_sents)

    tfidf_transformer = TfidfVectorizer(
        smooth_idf=True,
        use_idf=True,
        ngram_range=(1, 3),
        max_df=max_df,
        max_features=10000,
    )
    tfidf_transformer.fit(lemmatized_clusters)
    feature_names = tfidf_transformer.get_feature_names()

    keyword_clusters = {}

    for label, lemmatized_cluster in zip(labels, lemmatized_clusters):
        # generate tf-idf
        tf_idf_vector = tfidf_transformer.transform([lemmatized_cluster])
        # sort the tf-idf vectors by descending order of scores
        tuples = zip(tf_idf_vector.tocoo().col, tf_idf_vector.tocoo().data)
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        # extract only the top n
        keywords = extract_topn_from_vector(feature_names, sorted_items, topn * 10)
        keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)

        scores = {x[0]: x[1] for x in keywords}

        already_in = set()
        filtered_keywords = []
        for kw, score in keywords:
            if len(kw.split()) == 1:
                for k, s in scores.items():
                    if kw in k and len(k.split()) > 1:
                        if score > s:
                            already_in.add(k)
                        else:
                            already_in.add(kw)
            if len(kw.split()) == 2:
                for k, s in scores.items():
                    if kw in k and len(k.split()) > 2:
                        if score > s:
                            already_in.add(k)
                        else:
                            already_in.add(kw)

            if kw not in already_in and kw != target_word:
                filtered_keywords.append(kw)
                already_in.add(kw)

        keyword_clusters[label] = filtered_keywords[: topn * 10]

    final_keywords = {}
    all_data = []
    for c, keywords in keyword_clusters.items():
        sents = sent_clust_dict[c][0].split("\t")
        lemmas = sent_clust_dict[c][1].split("\t")
        all_sents = " ".join(sents)
        all_lemmas = " ".join(lemmas)
        set_lemmatized_sents = set(lemmas)
        filtered_keywords = []
        for kw in keywords:
            stop = 0
            for word in kw.split():
                if word in sw:
                    stop += 1
            if stop / float(len(kw.split())) < 0.5:
                num_appearances = 0
                for sent in set_lemmatized_sents:
                    if kw in sent:
                        num_appearances += 1
                if num_appearances > 1:
                    if len(kw) > 2:
                        if kw + " " + target_word in all_lemmas:
                            kw = kw + " " + target_word
                        elif target_word + " " + kw in all_lemmas:
                            kw = target_word + " " + kw
                        filtered_keywords.append(kw)
        if len(filtered_keywords) == 0:
            filtered_keywords.append("other")
        final_keywords[c] = filtered_keywords[:topn]
        all_data.append((c, ";".join(filtered_keywords[:50]), all_sents))
    return final_keywords


def full_analysis(
    word,
    labels,
    sentences,
    id2sent,
    corpus_slices,
    image_folder,
    docs_folder,
    max_df=0.8,
    topn=15,
    threshold_size_cluster=10,
    lang="slo",
):
    clusters_sents_df = get_clusters_sent(
        word,
        threshold_size_cluster,
        labels,
        sentences,
        id2sent,
        corpus_slices,
        docs_folder,
    )
    keyword_clusters = extract_keywords(
        word, clusters_sents_df, topn=topn, max_df=max_df, lang=lang
    )
    output_distrib(clusters_sents_df, word, keyword_clusters, image_folder)
    return keyword_clusters


def loadData(labels_path, sentences_path, id2sents_path):
    labels = dill.load(open(labels_path, "rb"))
    sentences = dill.load(open(sentences_path, "rb"))
    id2sents = dill.load(open(id2sents_path, "rb"))
    return labels, sentences, id2sents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpret changes")
    parser.add_argument(
        "--target_words",
        default="test",
        type=str,
        help="Target words to analyse, separated by comma",
    )
    parser.add_argument(
        "--lang",
        default="slo",
        type=str,
        help="Language of the corpus, currently only Slovenian ('slo') and English ('en') are supported",
    )
    parser.add_argument(
        "--input_dir",
        default="output",
        type=str,
        help="Folder containing data generated by the script 'measure_semantic_shift.py'",
    )
    parser.add_argument(
        "--results_dir", default="results", type=str, help="Path to final results"
    )
    parser.add_argument(
        "--max_df",
        type=float,
        default=0.8,
        help="Words that appear in more than that percentage of clusters will not be used as keywords.",
    )
    parser.add_argument(
        "--cluster_size_threshold",
        type=int,
        default=10,
        help="Clusters smaller than a threshold will be deleted.",
    )
    parser.add_argument(
        "--num_keywords", type=int, default=10, help="Number of keywords per cluster."
    )
    args = parser.parse_args()

    image_folder = os.path.join(args.results_dir, "images")
    docs_folder = os.path.join(args.results_dir, "docs")
    target_words = args.target_words.split(",")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(docs_folder, exist_ok=True)

    labels = os.path.join(args.input_dir, "kmeans_5_labels.pkl")
    sentences = os.path.join(args.input_dir, "sents.pkl")
    id2sent = os.path.join(args.input_dir, "id2sents.pkl")
    corpus_slices = os.path.join(args.input_dir, "corpus_slices.pkl")
    corpus_slices = dill.load(open(corpus_slices, "rb"))

    for word in target_words:
        print("Generating results for word:", word)
        keyword_clusters = full_analysis(
            word,
            labels,
            sentences,
            id2sent,
            corpus_slices,
            image_folder,
            docs_folder,
            max_df=args.max_df,
            topn=args.num_keywords,
            threshold_size_cluster=args.cluster_size_threshold,
            lang=args.lang,
        )
        print("Results written to folder:", args.results_dir)
        print("--------------------------------------------")
