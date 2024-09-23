import math
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class Indexer:
    def __init__(self):
        self.dictionary = {}
        self.postings = defaultdict(list)
        self.doc_lengths = {}
        self.num_docs = 0

        # Download required NLTK data
        # nltk.download("punkt", quiet=True)
        # nltk.download("stopwords", quiet=True)

        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def preprocess(self, text):
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove stop words and apply stemming
        return [
            self.stemmer.stem(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]

    def add_document(self, doc_id, content):
        self.num_docs += 1
        terms = self.preprocess(content)
        term_freq = defaultdict(int)

        for term in terms:
            term_freq[term] += 1

        doc_length = 0
        for term, freq in term_freq.items():
            if term not in self.dictionary:
                self.dictionary[term] = len(self.dictionary)

            log_tf = 1 + math.log10(freq)
            self.postings[term].append((doc_id, log_tf))
            doc_length += log_tf**2

        self.doc_lengths[doc_id] = math.sqrt(doc_length)

    def finalize_index(self):
        for term in self.dictionary:
            self.postings[term].sort(key=lambda x: x[0])  # Sort postings by doc_id

    def save_index(self, dictionary_file, postings_file, doc_lengths_file):
        with open(dictionary_file, "w") as f:
            for term, term_id in self.dictionary.items():
                f.write(f"{term},{term_id}\n")

        with open(postings_file, "w") as f:
            for term, postings in self.postings.items():
                postings_str = " ".join([f"{doc_id}:{tf}" for doc_id, tf in postings])
                f.write(f"{term},{postings_str}\n")

        with open(doc_lengths_file, "w") as f:
            for doc_id, length in self.doc_lengths.items():
                f.write(f"{doc_id},{length}\n")


# Usage example:
indexer = Indexer()
# Add documents
indexer.add_document(1, "This is the first document about information retrieval.")
indexer.add_document(
    2, "This document is the second document discussing search engines."
)
indexer.finalize_index()
indexer.save_index("dictionary.txt", "postings.txt", "doc_lengths.txt")
