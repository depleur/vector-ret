import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data if not already downloaded
nltk.download("punkt")
nltk.download("stopwords")

# Initialize required objects
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


# Preprocessing function
def preprocess(text):
    # Case folding: convert text to lowercase
    text = text.lower()

    # Normalization: remove punctuation and special characters
    text = re.sub(r"\W+", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stop word removal
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming (you can replace it with lemmatization if needed)
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

    return " ".join(stemmed_tokens)  # Join tokens back into a single string


# Process the entire corpus and save to a new directory
def process_and_save_corpus(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            # Read the original file
            with open(os.path.join(input_directory, filename), "r") as file:
                text = file.read()
                # Preprocess the text
                processed_text = preprocess(text)
                # Save the preprocessed text to a new file in the output directory
                with open(os.path.join(output_directory, filename), "w") as output_file:
                    output_file.write(processed_text)


# Paths to the input and output directories
input_directory = "./Corpus/"
output_directory = "./Corpus/preprocessed/"

# Preprocess and save the new corpus
process_and_save_corpus(input_directory, output_directory)

print(f"Preprocessed files have been saved to {output_directory}")
