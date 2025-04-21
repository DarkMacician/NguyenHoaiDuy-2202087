import string
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text, stop_words=stop_words):
    # Lowercase the text
    text = text.lower()
    #text = text.replace('"', '')

    # Remove double quotes from the start and end of words
    text = re.sub(r"(^['\"]|['\"]$)", '', text)
    # Remove other punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    for i in words:
        if '"' in i:
            print(i)

    return words

def read_and_clean_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    return clean_text(raw_text)

def plot_wordcloud(word_freq):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud")
    plt.savefig("wordcloud.png")
    plt.show()


def plot_barchart(word_freq, top_n=20):
    most_common = word_freq.most_common(top_n)
    words, counts = zip(*most_common)
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='skyblue')
    plt.xticks(rotation=45)
    plt.title(f"Top {top_n} Most Frequent Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("barchart.png")
    plt.show()


def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def plot_semantic_relationships(word_freq, glove_path, top_n=50):
    # Load GloVe embeddings
    embeddings = load_glove_embeddings(glove_path)

    # Get top frequent words and their vectors
    words, vectors = [], []
    for word, _ in word_freq.most_common(top_n):
        if word in embeddings:
            words.append(word)
            vectors.append(embeddings[word])

    if len(words) < 2:
        print("Not enough words found in GloVe to plot semantic relationships.")
        return

    # PCA to reduce dimensions
    reduced = PCA(n_components=2).fit_transform(vectors)

    # Plot the semantic relationship in 2D
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], color='orange')

    for i, word in enumerate(words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=11)

    plt.title("Semantic Word Relationships (GloVe + PCA)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("semantic_relationships.png")
    plt.show()


def plot_heatmap(word_freq, glove_path, top_n=10):
    # Load GloVe embeddings
    embeddings = load_glove_embeddings(glove_path)

    # Lọc ra các từ phổ biến nhất có tồn tại trong GloVe
    words_vectors = [(word, embeddings[word]) for word, _ in word_freq.most_common(top_n) if word in embeddings]

    if len(words_vectors) < 2:
        print("Not enough words found in GloVe to plot heatmap.")
        return

    words, vectors = zip(*words_vectors)

    cosine_sim = cosine_similarity(vectors)

    plt.figure(figsize=(12, 9))
    sns.heatmap(cosine_sim, xticklabels=words, yticklabels=words, cmap='coolwarm', annot=True, fmt=".2f", annot_kws={"size": 9})
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Word Similarities (Top {len(words)} Words)")
    plt.tight_layout()
    plt.savefig("heatmap.png")
    plt.show()


# === Run it ===
file_path = 'text.txt'
glove_path = 'glove.6B.100d.txt'

cleaned_words = read_and_clean_txt(file_path)
word_freq = Counter(cleaned_words)
print(word_freq.keys())

#Plot Word Cloud
plot_wordcloud(word_freq)

# Plot Bar Chart for top frequent words
plot_barchart(word_freq)

# Plot Semantic Word Relationships using PCA (reduce to 2D)
plot_semantic_relationships(word_freq, glove_path)

# Plot Word Similarities as Heatmap
plot_heatmap(word_freq, glove_path)