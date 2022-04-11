import dill
import pandas as pd
import ot
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
from scipy.stats import entropy
from collections import defaultdict
import numpy as np
import os
import argparse


def combine_clusters(labels, embeddings, treshold=10, remove=[]):
    cluster_embeds = defaultdict(list)
    for label, embed in zip(labels, embeddings):
        cluster_embeds[label].append(embed)
    min_num_examples = treshold
    legit_clusters = []
    for id, num_examples in Counter(labels).items():
        if num_examples >= treshold:
            legit_clusters.append(id)
        if id not in remove and num_examples < min_num_examples:
            min_num_examples = num_examples
            min_cluster_id = id

    if len(set(labels)) == 2:
        return labels

    min_dist = 1
    all_dist = []
    cluster_labels = ()
    embed_list = list(cluster_embeds.items())
    for i in range(len(embed_list)):
        for j in range(i+1,len(embed_list)):
            id, embed = embed_list[i]
            id2, embed2 = embed_list[j]
            if id in legit_clusters and id2 in legit_clusters:
                dist = compute_averaged_embedding_dist(embed, embed2)
                all_dist.append(dist)
                if dist < min_dist:
                    min_dist = dist
                    cluster_labels = (id, id2)

    std = np.std(all_dist)
    avg = np.mean(all_dist)
    limit = avg - 2 * std
    if min_dist < limit:
        for n, i in enumerate(labels):
            if i == cluster_labels[0]:
                labels[n] = cluster_labels[1]
        return combine_clusters(labels, embeddings, treshold, remove)

    if min_num_examples >= treshold:
        return labels


    min_dist = 2
    cluster_labels = ()
    for id, embed in cluster_embeds.items():
        if id != min_cluster_id:
            dist = compute_averaged_embedding_dist(embed, cluster_embeds[min_cluster_id])
            if dist < min_dist:
                min_dist = dist
                cluster_labels = (id, min_cluster_id)

    if cluster_labels[0] not in legit_clusters:
        for n, i in enumerate(labels):
            if i == cluster_labels[0]:
                labels[n] = cluster_labels[1]
    else:
        if min_dist < limit:
            for n, i in enumerate(labels):
                if i == cluster_labels[0]:
                    labels[n] = cluster_labels[1]
        else:
            remove.append(min_cluster_id)
    return combine_clusters(labels, embeddings, treshold, remove)


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def cluster_word_embeddings_k_means(word_embeddings, k, random_state):
    clustering = KMeans(n_clusters=k, random_state=random_state).fit(word_embeddings)
    labels = clustering.labels_
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def compute_averaged_embedding_dist(t1_embeddings, t2_embeddings):
    t1_mean = np.mean(t1_embeddings, axis=0)
    t2_mean = np.mean(t2_embeddings, axis=0)
    dist = 1.0 - cosine_similarity([t1_mean], [t2_mean])[0][0]
    return dist


def compute_divergence_from_cluster_labels(embeds1, embeds2, labels1, labels2, weights1, weights2, treshold):
    labels_all = list(np.concatenate((labels1, labels2)))
    n_senses = list(set(labels_all))
    counts1 = Counter(labels1)
    counts2 = Counter(labels2)

    t1 = defaultdict(int)
    for l, c in zip(labels1, weights1):
        if counts1[l] + counts2[l] > treshold:
            t1[l] += c
    t1 = sorted(t1.items(), key=lambda x: x[0])
    t1_dist = np.array([x[1] / sum(weights1) for x in t1])

    t2 = defaultdict(int)
    for l, c in zip(labels2, weights2):
        if counts1[l] + counts2[l] > treshold:
            t2[l] += c
    t2 = sorted(t2.items(), key=lambda x: x[0])
    t2_dist = np.array([x[1] / sum(weights2) for x in t2])
    label_list = sorted(list(n_senses))

    emb1_means = np.array([np.mean(embeds1[labels1 == clust], 0) for clust in label_list])
    emb2_means = np.array([np.mean(embeds2[labels2 == clust], 0) for clust in label_list])
    M = np.nan_to_num(np.array([cdist(emb1_means, emb2_means, metric='cosine')])[0], nan=1)

    wass = ot.emd2(t1_dist, t2_dist, M)
    jsd = compute_jsd(t1_dist, t2_dist)
    return jsd, wass


def detect_meaning_gain_and_loss(labels1, labels2, treshold):
    labels1 = list(labels1)
    labels2 = list(labels2)
    all_count = Counter(labels1 + labels2)
    first_count = Counter(labels1)
    second_count = Counter(labels2)
    gained_meaning = False
    lost_meaning = False
    all = 0
    meaning_gain_loss = 0

    for label, c in all_count.items():
        all += c
        if c >= treshold:
            if label not in first_count or first_count[label] <= 2:
                gained_meaning=True
                meaning_gain_loss += c
            if label not in second_count or second_count[label] <= 2:
                lost_meaning=True
                meaning_gain_loss += c
    return str(gained_meaning) + '/' + str(lost_meaning), meaning_gain_loss/all


def compute_divergence_across_many_periods(embeddings, counts, labels, splits, corpus_slices, treshold, metric):
    all_clusters = []
    all_embeddings = []
    all_counts = []
    clusters_dict = {}
    distrib_dict = {}
    for split_num, split in enumerate(splits):
        if split_num > 0:
            clusters = labels[splits[split_num-1]:split]
            clusters_dict[corpus_slices[split_num - 1]] = clusters
            all_clusters.append(clusters)
            ts_embeds = embeddings[splits[split_num - 1]:split]
            ts_counts = counts[splits[split_num - 1]:split]
            all_embeddings.append(ts_embeds)
            all_counts.append(ts_counts)
            distrib = defaultdict(int)
            for l, c in zip(clusters, ts_counts):
                distrib[l] += c
            distrib = sorted(distrib.items(), key=lambda x: x[0])
            distrib = np.array([x[1] / sum(ts_counts) for x in distrib])
            distrib_dict[corpus_slices[split_num - 1]] = distrib
    all_scores = []
    all_meanings = []
    for i in range(len(all_clusters)):
        if i < len(all_clusters) -1:
            try:
                jsd, wass = compute_divergence_from_cluster_labels(all_embeddings[i],all_embeddings[i+1], all_clusters[i],all_clusters[i+1], all_counts[i], all_counts[i+1], treshold)
            except:
                jsd, wass = 0, 0
            meaning, meaning_score = detect_meaning_gain_and_loss(all_clusters[i],all_clusters[i+1], treshold)
            all_meanings.append(meaning)
            try:
                if metric == 'JSD':
                    measure = jsd
                if metric == 'WD':
                    measure = wass
            except:
                measure = 0
            all_scores.append(measure)
    try:
        entire_jsd, entire_wass = compute_divergence_from_cluster_labels(all_embeddings[0],all_embeddings[-1], all_clusters[0],all_clusters[-1], all_counts[0], all_counts[-1], treshold)
    except:
        entire_jsd, entire_wass = 0, 0
    meaning, meaning_score = detect_meaning_gain_and_loss(all_clusters[0],all_clusters[-1], treshold)
    all_meanings.append(meaning)


    avg_score = sum(all_scores)/len(all_scores)
    try:
        if metric == 'JSD':
            measure = entire_jsd
        if metric == 'WD':
            measure = entire_wass
    except:
        measure = 0
    all_scores.extend([measure, avg_score])
    all_scores = [float("{:.6f}".format(score)) for score in all_scores]
    return all_scores, all_meanings, clusters_dict, distrib_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="Paths to output results folder")
    parser.add_argument("--embeddings_path",
                        default='embeddings/embeddings.pickle', type=str,
                        help="Path to input pickle file containing embeddings.")
    parser.add_argument("--random_state", default=123, type=int, help="Random seed.")
    parser.add_argument("--cluster_size_threshold", default=10, type=int, help="Remove cluster or merge it with other if smaller than threshold")
    parser.add_argument("--metric",
                        default='JSD',
                        type=str,
                        help="Which metric to use for measuring semantic shift, should be JSD or WD")
    args = parser.parse_args()

    metric = args.metric
    assert metric in ['JSD', 'WD']
    embeddings_path = args.embeddings_path
    output_dir = args.output_dir

    embeddings, count2sents = dill.load(open(embeddings_path, 'rb'))
    id2sent = {}

    target_words = list(embeddings.keys())

    corpus_slices = []
    for w in target_words[:500]:
        for cs in embeddings[w].keys():
            corpus_slices.append(cs)
    corpus_slices = sorted(list(set([x for x in corpus_slices if not x.endswith('_text')])))
    print("Corpus slices: ", corpus_slices)

    sentence_dict = {}
    kmeans_5_labels_dict = {}
    kmeans_5_centroids_dict = {}

    print("Clustering BERT embeddings")
    print("Len target words: ", len(target_words))

    results = []

    print("Num. words in embeds: ", len(embeddings.keys()))

    for i, word in enumerate(target_words):
        print("\n=======", i + 1, "- word:", word.upper(), "=======")

        if word not in embeddings:
            continue
        emb = embeddings[word]

        all_embeddings = []
        all_sentences = {}
        all_counts = []
        splits = [0]
        all_slices_present = True
        all_freqs = []

        summed_cs_counts = []

        for cs in corpus_slices:
            cs_embeddings = []
            cs_sentences = []
            cs_counts = []

            count_all = 0
            text_seen = set()

            if cs not in emb:
                all_slices_present = False
                print('Word missing in slice: ', cs)
                continue

            counts = [x[1] for x in emb[cs]]
            summed_cs_counts.append(sum(counts))
            #print('Counts: ', counts)
            all_freqs.append(sum(counts))
            cs_text = cs + '_text'
            print("Slice: ", cs)
            print("Num embeds: ", len(emb[cs]))
            num_sent_codes = 0

            for idx in range(len(emb[cs])):

                #get summed embedding and its count, devide embedding by count
                try:
                    e, count_emb = emb[cs][idx]
                    e = e/count_emb
                except:
                    e = emb[cs][idx]

                sents = set()

                if count2sents is not None:
                    sent_codes = emb[cs_text][idx]
                    num_sent_codes += len(sent_codes)
                    for sent in sent_codes:
                        if sent in count2sents[cs]:
                            sent_data = count2sents[cs][sent]
                        sent_id = int(str(corpus_slices.index(cs) + 1) + str(sent))
                        id2sent[sent_id] = sent_data
                        sents.add(sent)

                cs_embeddings.append(e)
                cs_sentences.append(sents)
                cs_counts.append(count_emb)

            all_embeddings.append(np.array(cs_embeddings))
            all_sentences[cs] = cs_sentences
            all_counts.append(np.array(cs_counts))
            splits.append(splits[-1] + len(cs_embeddings))


        print("Num all sents: ", num_sent_codes)
        print("Num words in corpus slice: ", summed_cs_counts)

        embeddings_concat = np.concatenate(all_embeddings, axis=0)
        counts_concat = np.concatenate(all_counts, axis=0)

        if embeddings_concat.shape[0] < 5 or not all_slices_present:
            continue
        else:
            kmeans_5_labels, kmeans_5_centroids = cluster_word_embeddings_k_means(embeddings_concat, 5, args.random_state)
            kmeans_5_labels = combine_clusters(kmeans_5_labels, embeddings_concat, treshold=args.cluster_size_threshold, remove=[])
            all_kmeans5_jsds, all_meanings, clustered_kmeans_5_labels, distrib_kmeans_5 = compute_divergence_across_many_periods(embeddings_concat, counts_concat, kmeans_5_labels, splits, corpus_slices, args.cluster_size_threshold, metric)
            all_freqs = all_freqs + [sum(all_freqs)] + [sum(all_freqs)/len(all_freqs)]
            word_results = [word] + all_kmeans5_jsds + all_freqs + all_meanings
            print("Results:", word_results)
            results.append(word_results)

        sentence_dict[word] = all_sentences
        kmeans_5_labels_dict[word] = clustered_kmeans_5_labels
        kmeans_5_centroids_dict[word] = kmeans_5_centroids

    columns = ['word']
    methods = [metric + ' K5', 'FREQ', 'MEANING GAIN/LOSS']
    for method in methods:
        for num_slice, cs in enumerate(corpus_slices):
            if method == 'FREQ':
                columns.append(method + ' ' + cs)
            else:
                if num_slice < len(corpus_slices) - 1:
                    columns.append(method + ' ' + cs + '-' + corpus_slices[num_slice + 1])
        columns.append(method + ' All')
        if method != 'MEANING GAIN/LOSS':
            columns.append(method + ' Avg')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_file = os.path.join(output_dir, "word_list_results.csv")

    # save results to CSV
    results_df = pd.DataFrame(results, columns=columns)
    results_df = results_df.sort_values(by=[metric + ' K5 Avg'], ascending=False)
    results_df.to_csv(csv_file, sep=';', encoding='utf-8', index=False)

    print("Done! Saved results in", csv_file, "!")

    # save cluster labels and sentences to pickle
    dicts = [(kmeans_5_centroids_dict, 'kmeans_5_centroids'), (kmeans_5_labels_dict, 'kmeans_5_labels'),
             (sentence_dict, "sents"), (id2sent, "id2sents")]

    for data, name in dicts:
        data_file = os.path.join(output_dir, name + ".pkl")
        pf = open(data_file, 'wb')
        dill.dump(data, pf)
        pf.close()

    data_file = os.path.join(output_dir, "corpus_slices.pkl")
    pf = open(data_file, 'wb')
    dill.dump(corpus_slices, pf)
    pf.close()
