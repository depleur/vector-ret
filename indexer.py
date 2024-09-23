import math
from collections import defaultdict
import os


class Indexer:
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.dictionary = {}
        self.postings = defaultdict(list)
        self.doc_lengths = {}
        self.num_docs = 0
        self.doc_id_map = {}  # Maps company names to numeric doc_ids

    def index_corpus(self):
        for filename in os.listdir(self.corpus_dir):
            if filename.endswith(".txt"):
                company_name = filename[:-4]  # Remove '.txt' extension
                doc_id = len(self.doc_id_map) + 1
                self.doc_id_map[company_name] = doc_id

                with open(os.path.join(self.corpus_dir, filename), "r") as f:
                    preprocessed_terms = (
                        f.read().split()
                    )  # Assuming terms are space-separated
                self.add_document(doc_id, preprocessed_terms)

        self.finalize_index()

    def add_document(self, doc_id, preprocessed_terms):
        self.num_docs += 1
        term_freq = defaultdict(int)

        for term in preprocessed_terms:
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

    def save_index(
        self, dictionary_file, postings_file, doc_lengths_file, doc_id_map_file
    ):
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

        with open(doc_id_map_file, "w") as f:
            for company, doc_id in self.doc_id_map.items():
                f.write(f"{company},{doc_id}\n")


# Usage example:
indexer = Indexer("./preprocessed_corpus/")
indexer.index_corpus()
indexer.save_index(
    "dictionary.txt", "postings.txt", "doc_lengths.txt", "doc_id_map.txt"
)
