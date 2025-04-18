import nltk
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')

# Function to perform extractive summarization
def extractive_summarization(text, num_sentences):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return "The text is too short for summarization!"

    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)

    similarity_matrix = np.dot(sentence_vectors, sentence_vectors.T).toarray()

    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [ranked_sentences[i][1] for i in range(min(num_sentences, len(sentences)))]
    
    return " ".join(summary_sentences)
