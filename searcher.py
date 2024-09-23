import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class Searcher:
    def __init__(self, dictionary_file, postings_file, doc_lengths_file):
        self.dictionary = {}
        self.postings = {}
        self.doc_lengths = {}
        self.num_docs = 0

        # Download required NLTK data
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)

        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

        self.load_index(dictionary_file, postings_file, doc_lengths_file)

    def preprocess(self, text):
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove stop words and apply stemming
        return [
            self.stemmer.stem(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]

    def load_index(self, dictionary_file, postings_file, doc_lengths_file):
        with open(dictionary_file, "r") as f:
            for line in f:
                term, term_id = line.strip().split(",")
                self.dictionary[term] = int(term_id)

        with open(postings_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                term = parts[0]
                postings = [tuple(map(float, p.split(":"))) for p in parts[1].split()]
                self.postings[term] = postings

        with open(doc_lengths_file, "r") as f:
            for line in f:
                doc_id, length = line.strip().split(",")
                self.doc_lengths[int(doc_id)] = float(length)

        self.num_docs = len(self.doc_lengths)

    def search(self, query):
        query_terms = self.preprocess(query)
        query_weights = self.compute_query_weights(query_terms)

        scores = defaultdict(float)
        for term, weight in query_weights.items():
            if term in self.postings:
                idf = math.log10(self.num_docs / len(self.postings[term]))
                for doc_id, tf in self.postings[term]:
                    scores[doc_id] += weight * idf * tf

        # Normalize scores
        for doc_id in scores:
            scores[doc_id] /= self.doc_lengths[doc_id]

        # Sort results
        results = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [doc_id for doc_id, score in results[:10]]

    def compute_query_weights(self, query_terms):
        tf = defaultdict(int)
        for term in query_terms:
            tf[term] += 1

        weights = {}
        query_length = 0
        for term, freq in tf.items():
            weight = (1 + math.log10(freq)) * math.log10(
                self.num_docs / len(self.postings.get(term, []))
            )
            weights[term] = weight
            query_length += weight**2

        query_length = math.sqrt(query_length)
        for term in weights:
            weights[term] /= query_length

        return weights


# Usage example:
searcher = Searcher("dictionary.txt", "postings.txt", "doc_lengths.txt")
results = searcher.search("information retrieval and search engines")
print(results)
