import argparse
import gc
import pickle
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForMaskedLM, AutoTokenizer


BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

# BOS_TOKEN='[CLS]'
# EOS_TOKEN='[SEP]'


def chunks(lst, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def add_embedding_to_list(previous, word_emb):
    embeds = [x[0] / x[1] for x in previous]
    cs = list(cosine_similarity(word_emb.reshape(1, -1), np.array(embeds))[0].tolist())
    if len(previous) < 200 and max(cs) < 0.99:
        # if len(previous) < 1:
        max_idx = len(previous)
        previous.append((word_emb, 1))
    else:
        max_idx = cs.index(max(cs))
        old_embd, count = previous[max_idx]
        new_embd = old_embd + word_emb
        count = count + 1
        previous[max_idx] = (new_embd, count)
    return previous, max_idx


def addPosition(sent_data, length):
    mapping, sent, lemma = sent_data
    new_mapping = []
    for word, idxs in mapping:
        new_idxs = [i + length for i in idxs]
        new_mapping.append((word, new_idxs))
    return (new_mapping, sent, lemma)


def mapSent2lemmaRoberta(sent, lemma, sent_id):
    conjunctives = ["▁-"]
    lemma = lemma.split()
    mapping = []
    count = 0
    idxs = []
    all_idxs = []
    for idx, token in enumerate(sent):
        if token in [BOS_TOKEN]:
            continue
        if token in [EOS_TOKEN]:
            if len(idxs) > 0:
                try:
                    mapping.append((lemma[count], idxs))
                except:
                    print("Error in mapping")
                    print(sent)
                    print(lemma)
                    print("----------------------")
                    return None
                all_idxs.extend(idxs)
                idxs = []

        elif sent[idx + 1] in conjunctives or not sent[idx + 1].startswith("▁"):
            idxs.append(idx)
        elif token in conjunctives:
            idxs.append(idx)
        elif sent[idx + 1] not in conjunctives and sent[idx + 1].startswith("▁"):
            idxs.append(idx)
            all_idxs.extend(idxs)
            try:
                mapping.append((lemma[count], idxs))
            except:
                print("Error in mapping")
                print(sent)
                print(lemma)
                print("----------------------")
                return None

            count += 1
            idxs = []
    length_sent = len(sent)
    length_mapping = len(all_idxs)
    if length_sent - 2 != length_mapping:
        print("Wrong length")
        print(sent)
        print(lemma)
        print("----------------------")
        return None
    return (mapping, sent_id, lemma)


def mapSent2lemma(sent, lemma, sent_id):
    conjunctives = ["-"]
    lemma = lemma.split()
    mapping = []
    count = 0
    idxs = []
    all_idxs = []
    for idx, token in enumerate(sent):
        if token in [EOS_TOKEN, BOS_TOKEN]:
            continue
        elif sent[idx + 1] in conjunctives or sent[idx + 1].startswith("##"):
            idxs.append(idx)
        elif token in conjunctives:
            idxs.append(idx)
        elif sent[idx + 1] not in conjunctives and not sent[idx + 1].startswith("##"):
            idxs.append(idx)
            all_idxs.extend(idxs)
            try:
                mapping.append((lemma[count], idxs))
            except:
                print("Error in mapping")
                print(sent)
                print(lemma)
                print("----------------------")
                return None

            count += 1
            idxs = []
    length_sent = len(sent)
    length_mapping = len(all_idxs)
    if length_sent - 2 != length_mapping:
        print("Wrong length")
        print(sent)
        print(lemma)
        print("----------------------")
        return None
    return (mapping, sent_id, lemma)


def tokens_to_batches(ds, tokenizer, batch_size, max_length):
    batches = []
    batch = []
    batch_counter = 0

    counter = 0
    sent_counter = 0
    count2sent = {}
    all_vocab = set()
    prev_vocab = 0
    error_counter = 0

    for doc, lemmatized_doc, meta in ds:
        sents = doc.split(" <eos> ")
        lemmatized_sents = lemmatized_doc.split(" <eos> ")
        tokenized_text = []
        seq_mappings = []

        for sent, lemmatized_sent in zip(sents, lemmatized_sents):
            sent = sent.replace("&amp;", "&")
            sent = sent.replace("&lt;", "lt")
            sent = sent.replace("&gt;", "gt")
            counter += 1

            if counter % 500000 == 0:
                print("Num sentences: ", counter)
            if (
                len(all_vocab) % 100000 == 0
                and len(all_vocab) > 0
                and len(all_vocab) > prev_vocab
            ):
                prev_vocab = len(all_vocab)
                print("Vocab size: ", len(all_vocab))

            sent_counter += 1
            lsent = sent.strip()
            if len(lsent.split()) > 2:
                marked_sent = f"{BOS_TOKEN} {lsent} {EOS_TOKEN}"
                tokenized_sent = tokenizer.tokenize(marked_sent)
                if len(tokenized_sent) > max_length:
                    continue
                tokenized_lemma = tokenizer.tokenize(lemmatized_sent)
                lemmatized_sent = tokenizer.convert_tokens_to_string(tokenized_lemma)
                if BOS_TOKEN == "<s>":
                    mapping = mapSent2lemmaRoberta(
                        tokenized_sent, lemmatized_sent, sent_counter
                    )
                else:
                    mapping = mapSent2lemma(
                        tokenized_sent, lemmatized_sent, sent_counter
                    )
                if mapping is None:
                    error_counter += 1
                    continue
                lemmas = mapping[2]
                for l in lemmas:
                    all_vocab.add(l)

                count2sent[sent_counter] = (sent, lemmatized_sent, meta)

                if len(tokenized_text) + len(tokenized_sent) > max_length:
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    # print("Batch counter: ", len(tokenized_text), batch_counter, tokenized_text)
                    batch.append((indexed_tokens, seq_mappings, tokenized_text))
                    # print(tokenized_text)
                    # print(seq_mappings)
                    batch_counter += 1
                    tokenized_text = tokenized_sent
                    seq_mappings = [mapping]
                    if batch_counter % batch_size == 0:
                        batches.append(batch)
                        batch = []
                else:
                    mapping = addPosition(mapping, len(tokenized_text))
                    seq_mappings.append(mapping)
                    tokenized_text.extend(tokenized_sent)

        if len(tokenized_text) > 0:
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            batch.append((indexed_tokens, seq_mappings, tokenized_text))
            batch_counter += 1
            if batch_counter % batch_size == 0:
                batches.append(batch)
                batch = []

    if batch_counter % batch_size != 0:
        batches.append(batch)

    print()
    print("Tokenization done!")
    print("Num. wrong mappings:", error_counter)
    print("Final Vocab size: ", len(all_vocab))
    print("len batches: ", len(batches))

    return batches, count2sent


def get_token_embeddings(batches, model, batch_size, device):

    encoder_token_embeddings = []
    mappings = []
    tokens = []
    counter = 0

    for batch in batches:
        counter += 1
        if counter % 1000 == 0:
            print("Generating embedding for batch: ", counter)
        lens = [len(x[0]) for x in batch]
        max_len = max(lens)
        tokens_tensor = torch.zeros(batch_size, max_len, dtype=torch.long).to(device)
        segments_tensors = torch.ones(batch_size, max_len, dtype=torch.long).to(device)
        batch_idx = [x[0] for x in batch]
        batch_mapping = [x[1] for x in batch]
        batch_text = [x[2] for x in batch]

        for i in range(len(batch)):
            length = len(batch_idx[i])
            for j in range(max_len):
                if j < length:
                    tokens_tensor[i][j] = batch_idx[i][j]

        # print("Input shape: ", tokens_tensor.shape)
        # print(tokens_tensor)

        # Predict hidden states features for each layer
        with torch.no_grad():
            model_output = model(tokens_tensor, segments_tensors)
            encoded_layers = model_output[-1][-4:]  # last four layers of the encoder

        for batch_i in range(len(batch)):
            encoder_token_embeddings_example = []
            # For each token in the sentence...
            for token_i in range(len(batch_idx[batch_i])):

                # Holds 12 layers of hidden states for each token
                hidden_layers = []

                # For each of the 12 layers...
                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][batch_i][token_i]
                    hidden_layers.append(vec)

                hidden_layers = (
                    torch.sum(torch.stack(hidden_layers)[-4:], 0)
                    .reshape(1, -1)
                    .detach()
                    .cpu()
                    .numpy()
                )

                encoder_token_embeddings_example.append(hidden_layers)

            encoder_token_embeddings.append(encoder_token_embeddings_example)
            mappings.append(batch_mapping[batch_i])
            tokens.append(batch_text[batch_i])

        # Sanity check the dimensions:
        # print("Number of tokens in sequence:", len(token_embeddings))
        # print("Number of layers per token:", len(token_embeddings[0]))

    return encoder_token_embeddings, mappings, tokens


def combine_bpe(idxs, context_embedding, tokens):
    if len(idxs) == 1:
        encoder_array = context_embedding[idxs[0]]
    else:
        if BOS_TOKEN == "<s>":
            lengths = [len(tokens[i].replace("▁", "")) for i in idxs]
        else:
            lengths = [len(tokens[i].replace("##", "")) for i in idxs]
        weights = [x / sum(lengths) for x in lengths]

        encoder_array = np.zeros((1, 768))
        for i, weight in zip(idxs, weights):
            encoder_array += context_embedding[i] * weight
    return encoder_array


def get_slice_embeddings(
    embeddings_path, vocab, tokenizer, model, batch_size, max_length, device
):
    vocab_vectors = defaultdict(dict)
    count2sents = {}
    not_in_targets = set()
    print("All vocab chunks: ", vocab.chunks)
    targets = set(x[0] for x in vocab.freqs)

    for chunk in vocab.chunks:
        chunk = str(chunk)
        print("WORKING ON SLICE: ", chunk)
        ds = zip(vocab.docs[chunk], vocab.lemmatized_docs[chunk], vocab.meta[chunk])
        all_batches, count2sent = tokens_to_batches(
            ds, tokenizer, batch_size, max_length
        )

        count2sents[chunk] = count2sent
        chunked_batches = chunks(all_batches, 1000)
        num_chunk = 0

        for batches in chunked_batches:
            num_chunk += 1
            print("Chunk ", num_chunk)

            # get list of embeddings and list of bpe tokens
            encoder_token_embeddings, mappings, tokens = get_token_embeddings(
                batches, model, batch_size, device
            )

            # go through text token by token
            for emb_idx, ctx_embedding in enumerate(encoder_token_embeddings):
                seq_mappings = mappings[emb_idx]
                for mapping, sent_id, lemma_sent in seq_mappings:
                    sent_tokens = []
                    for token_i, ixs in mapping:
                        if token_i not in targets:
                            not_in_targets.add(token_i)
                            continue

                        encoder_array = combine_bpe(ixs, ctx_embedding, tokens[emb_idx])
                        encoder_array = encoder_array.squeeze()
                        token_chunks = vocab_vectors[token_i]
                        if chunk in token_chunks:
                            previous = token_chunks[chunk]
                            new, new_idx = add_embedding_to_list(
                                previous, encoder_array
                            )
                            token_chunks[chunk] = new
                            sent_tokens.append((token_i, new_idx))
                        else:
                            token_chunks[chunk] = [(encoder_array, 1)]
                            token_chunks[chunk + "_text"] = defaultdict(list)
                            sent_tokens.append((token_i, 0))

                    for sent_token, sent_idx in sent_tokens:
                        vocab_vectors[sent_token][chunk + "_text"][sent_idx].append(
                            sent_id
                        )

            del encoder_token_embeddings
            del batches
            gc.collect()

            """for k, v in vocab_vectors.items():
                print(k)
                input = v[0]
                encoder = v[1]
                context = v[2]
                print(len(input))
                print(len(encoder))
                print(len(context))
                print(context[0])"""

        print("Sentence embeddings generated.")
        print("Tokens not in targets: ", len(not_in_targets))

    print("Length of vocab after training: ", len(vocab_vectors.items()))

    with open(embeddings_path, "wb") as outfile:
        pickle.dump([vocab_vectors, count2sents], outfile)

    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab_path",
        default="output/vocab.pickle",
        type=str,
        help="Paths to vocab pickle file generated by the preprocessing.py script",
    )
    parser.add_argument(
        "--embeddings_path",
        default="embeddings/embeddings.pickle",
        type=str,
        help="Path to output pickle file containing embeddings.",
    )
    parser.add_argument(
        "--lang",
        default="slo",
        type=str,
        help="Language of the corpus, currently only Slovenian ('slo') and English ('en') are supported",
    )
    parser.add_argument(
        "--path_to_fine_tuned_model",
        default="",
        type=str,
        help="Path to fine-tuned model. If empty, pretrained model is used",
    )
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--max_sequence_length", default=256, type=int)
    parser.add_argument("--device", default="cuda", type=str, help="Which gpu to use")
    args = parser.parse_args()

    batch_size = args.batch_size
    max_length = args.max_sequence_length
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    if args.path_to_fine_tuned_model:
        model = AutoModelForMaskedLM.from_pretrained(
            args.path_to_fine_tuned_model, output_hidden_states=True
        )
        tokenizer = vocab.w_tokenizer
        print("Using a fined-tuned model", args.path_to_fine_tuned_model)
    else:
        print("Using a pretrained non-fined-tuned model.")
        assert args.lang in ["en", "slo"]
        if args.lang == "slo":
            model = AutoModelForMaskedLM.from_pretrained(
                "EMBEDDIA/sloberta", output_hidden_states=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "EMBEDDIA/sloberta", use_fast=False
            )
        elif args.lang == "en":
            model = AutoModelForMaskedLM.from_pretrained(
                "roberta-base", output_hidden_states=True
            )
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model.to(device)
    get_slice_embeddings(
        args.embeddings_path, vocab, tokenizer, model, batch_size, max_length, device
    )
