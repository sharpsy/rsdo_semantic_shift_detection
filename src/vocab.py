from collections import Counter, defaultdict


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

    def _iter_words_for_chunk(self, chunk):
        for doc in self.lemmatized_docs[chunk]:
            for sent in doc.split(" <eos> "):
                yield from sent.split()

    def make_vocab(self, vocab_path, lang, min_freq, stopwords):
        print("making_vocab")
        # words that do not have at least min_freq occurences in *every* split
        rare_words = set()
        bag_of_words = Counter()
        punctuation = frozenset("!#%'()*+,.:;=?@[\]^`{|}~")
        for chunk in self.chunks:
            print("chunk: ", chunk)
            chunk_bow = Counter()
            for word in self._iter_words_for_chunk(chunk):
                if punctuation.intersection(word) or word.isdigit():
                    continue
                if len(word) <= 2 or word.lower() in stopwords:
                    continue
                chunk_bow[word] += 1
            rare_words.update(wrd for wrd, cnt in chunk_bow.items() if cnt < min_freq)
            bag_of_words.update(chunk_bow)
        print("All vocab size: ", bag_of_words.total())

        # drop rare words
        for rare_word in rare_words:
            del bag_of_words[rare_word]

        print("Length of filtered vocabulary: ", bag_of_words.total())
        self.freqs = []
        with open(vocab_path, "w", encoding="utf8") as f:
            f.write("word,frequency\n")
            for w, freq in bag_of_words.most_common():
                w = self.w_tokenizer.tokenize(w)
                # w = "".join(w).replace('##', '')
                w = "".join(w).replace("▁", " ").strip()
                f.write(w + "," + str(freq) + "\n")
                self.freqs.append((w, freq))
