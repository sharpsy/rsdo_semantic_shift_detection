from collections import defaultdict


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

    def make_vocab(self, vocab_path, lang, min_freq, stopwords):
        print("making_vocab")
        all_freqs = []
        freqs = defaultdict(int)
        punctuation = "!#%'()*+,.:;=?@[\]^`{|}~"
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
                                if len(word) > 2 and word.lower() not in stopwords:
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
