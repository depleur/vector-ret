import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


class Searcher:
    def __init__(self, posting_list_file, doc_lengths_file):
        self.posting_list = {}
        self.doc_lengths = {}
        self.document_frequencies = {}
        self.num_docs = 0
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = nltk.stem.PorterStemmer()

        self.load_index(posting_list_file, doc_lengths_file)

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"\W+", " ", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        tokens = [self.stemmer.stem(word) for word in tokens]
        return tokens

    def load_index(self, posting_list_file, doc_lengths_file):
        with open(posting_list_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(": ")
                word_df = parts[0].split(", ")
                word = word_df[0].strip('"')
                df = int(word_df[1])
                self.document_frequencies[word] = df
                postings = eval(parts[1])
                self.posting_list[word] = {doc: float(tf) for doc, tf in postings}

        with open(doc_lengths_file, "r", encoding="utf-8") as f:
            for line in f:
                doc_name, length = line.strip().split(",")
                self.doc_lengths[doc_name] = float(length)

        self.num_docs = len(self.doc_lengths)

    def calculate_idf_weight(self, df):
        return math.log10(self.num_docs / df) if df > 0 else 0

    def normalize_vector(self, vector):
        norm = math.sqrt(sum(weight**2 for weight in vector.values()))
        return (
            {term: weight / norm for term, weight in vector.items()}
            if norm > 0
            else vector
        )

    def compute_cosine_similarity(self, query_vector, doc_vector):
        return sum(
            query_vector[term] * doc_vector.get(term, 0) for term in query_vector
        )

    def search(self, query):
        query_tokens = self.preprocess(query)
        query_vector = {}

        for word in set(query_tokens):
            tf = query_tokens.count(word)
            df = self.document_frequencies.get(word, 0)
            if df > 0:
                tf_weight = 1 + math.log10(tf)
                idf = self.calculate_idf_weight(df)
                query_vector[word] = tf_weight * idf

        query_vector = self.normalize_vector(query_vector)

        similarities = {}
        for doc_name in self.doc_lengths.keys():
            doc_vector = {}
            for word in query_vector:
                if word in self.posting_list and doc_name in self.posting_list[word]:
                    doc_vector[word] = self.posting_list[word][doc_name]

            doc_vector = self.normalize_vector(doc_vector)
            similarity = self.compute_cosine_similarity(query_vector, doc_vector)
            similarities[doc_name] = similarity

        ranked_docs = sorted(similarities.items(), key=lambda item: (-item[1], item[0]))
        return ranked_docs[:10]


# Usage
searcher = Searcher("posting_list.txt", "doc_lengths.txt")

query1 = "Developing your Zomato business account and profile is a great way to boost your restaurant's online reputation"
results1 = searcher.search(query1)
print("Ranked Documents by Relevance for query 1:")
for doc_name, score in results1:
    print(f"{doc_name}: {score:.4f}")

print("\n")

query2 = "Warwickshire, came from an ancient family and was the heiress to some land"
results2 = searcher.search(query2)
print("Ranked Documents by Relevance for query 2:")
for doc_name, score in results2:
    print(f"{doc_name}: {score:.4f}")
