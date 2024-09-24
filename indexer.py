import os
import math
from collections import defaultdict


class Indexer:
    def __init__(self, corpus_dir):
        # Step 1: Initialize indexer
        self.corpus_dir = corpus_dir
        self.documents = defaultdict(list)
        self.unique_words = set()
        self.posting_list = {}
        self.document_frequencies = defaultdict(int)
        self.doc_lengths = {}

    def read_documents(self):
        # Step 2: Read preprocessed documents
        for filename in os.listdir(self.corpus_dir):
            if filename.endswith(".txt"):
                with open(
                    os.path.join(self.corpus_dir, filename), "r", encoding="utf-8"
                ) as file:
                    text_tokens = (
                        file.read().split()
                    )  # Assuming tokens are space-separated
                    self.documents[filename] = text_tokens
                    self.unique_words.update(text_tokens)

    def create_posting_list(self):
        # Step 3: Create posting list and calculate document frequencies
        for word in self.unique_words:
            self.posting_list[word] = []
            for doc_name, text_tokens in self.documents.items():
                term_freq = text_tokens.count(word)
                if term_freq > 0:
                    self.document_frequencies[word] += 1
                    tf_weight = 1 + math.log10(term_freq)
                    self.posting_list[word].append((doc_name, tf_weight))

        # Step 4: Calculate document lengths
        for doc_name, text_tokens in self.documents.items():
            length = 0
            for word in set(text_tokens):
                tf = text_tokens.count(word)
                tf_weight = 1 + math.log10(tf)
                length += tf_weight**2
            self.doc_lengths[doc_name] = math.sqrt(length)

    def save_index(self, posting_list_file, doc_lengths_file):
        # Step 5: Save posting list and document lengths
        with open(posting_list_file, "w", encoding="utf-8") as f:
            for word, postings in self.posting_list.items():
                df = self.document_frequencies[word]
                f.write(f'"{word}", {df}: [')
                f.write(", ".join([f'("{doc}", "{tf:.6f}")' for doc, tf in postings]))
                f.write("]\n")

        with open(doc_lengths_file, "w", encoding="utf-8") as f:
            for doc_name, length in self.doc_lengths.items():
                f.write(f"{doc_name},{length}\n")

    def index_corpus(self):
        # Step 6: Execute indexing process
        self.read_documents()
        self.create_posting_list()
        self.save_index("posting_list.txt", "doc_lengths.txt")


# Usage
indexer = Indexer("./preprocessed_corpus/")
indexer.index_corpus()
