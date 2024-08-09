import logging

import json
from pprint import pprint

from collections import Counter

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, silhouette_samples

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import seaborn as sns


logging.basicConfig(level=logging.INFO)


def load_data(path):
    with open(path, 'r') as file:
        data = file.readlines()

        file.close()

    data = [json.loads(line) for line in data]
    data = [(example['statement'], example['verdict']) for example in data]

    return data


def encode_data(encoder, data):
    logging.info('Encoding data...')

    sentences, labels = zip(*data)
    sentences = encoder.encode(sentences)
    data = list(zip(sentences, labels))

    return data


def split_data(encoded_data, test_size=0.1):
    X, y = zip(*encoded_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

def train_model(data, label_counts, test_size=0.4):
    logging.info('Training model...')

    X_train, _, y_train, _ = data
    
    smote = SMOTE()
    
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_classifier, X_metamodel, y_classifier, y_metamodel = train_test_split(X_train, y_train, test_size=test_size)
    
    model = RandomForestClassifier()
    model.fit(X_classifier, y_classifier)
    
    y_pred = model.predict(X_metamodel)
    y_correct = [y_pred[idx] == y_metamodel[idx] for idx in range(len(y_metamodel))]
    
    metamodel = RandomForestClassifier()
    metamodel.fit(X_metamodel, y_correct)

    return model, metamodel



def filter_data(model, metamodel, data):
    logging.info('Filtering data...')

    X, y = data
    X_valid, y_valid = [], []

    for idx in range(len(X)):
        if metamodel.predict([X[idx]]) == 1:
            X_valid.append(X[idx])
            y_valid.append(y[idx])

    return X_valid, y_valid


def plot_label_distribution(labels, filename):
    label_counts = Counter(labels)

    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Count')

    plt.savefig(filename)
    plt.close()


def plot_sentence_length_distribution(sentences, filename):
    sentence_lengths = [len(sentence.split()) for sentence in sentences]

    plt.figure(figsize=(10, 6))
    plt.hist(sentence_lengths, bins=50)
    plt.title('Sentence Length Distribution')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')

    plt.savefig(filename)
    plt.close()


def plot_embedding_space(embeddings, labels, filename, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2)
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    label_to_num = {label: idx for idx, label in enumerate(set(labels))}
    numeric_labels = [label_to_num[label] for label in labels]
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=numeric_labels, cmap='viridis', alpha=0.5)

    plt.colorbar(scatter, ticks=range(len(label_to_num)), label='Labels')
    plt.clim(-0.5, len(label_to_num) - 0.5)
    plt.title(f'Embedding Space Visualization ({method.upper()})')

    plt.savefig(filename)
    plt.close()


def plot_class_balance(before_counts, after_counts, filename):
    labels = list(before_counts.keys())
    before_values = list(before_counts.values())
    after_values = list(after_counts.values())

    x = np.arange(len(labels))

    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, before_values, width, label='Before SMOTE')
    ax.bar(x + width/2, after_values, width, label='After SMOTE')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Count')
    ax.set_title('Class Balance Before and After SMOTE')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    plt.savefig(filename)

    plt.close()


def plot_embedding_density(embeddings, filename):
    plt.figure(figsize=(10, 6))

    sns.kdeplot(x=embeddings[:, 0], y=embeddings[:, 1], cmap='viridis', shade=True, bw_adjust=.5)

    plt.title('Embedding Density Plot')

    plt.savefig(filename)
    plt.close()


def plot_pairwise_distance_heatmap(embeddings, filename):
    distances = pairwise_distances(embeddings)

    plt.figure(figsize=(10, 6))

    sns.heatmap(distances, cmap='viridis')
    plt.title('Pairwise Distance Heatmap')

    plt.savefig(filename)
    plt.close()


def plot_silhouette_score(embeddings, labels, filename):
    silhouette_avg = silhouette_score(embeddings, labels)
    sample_silhouette_values = silhouette_samples(embeddings, labels)

    plt.figure(figsize=(10, 6))

    y_lower = 10

    for i in np.unique(labels):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        y_upper = y_lower + size_cluster_i

        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values)

        y_lower = y_upper + 10

    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    plt.title('Silhouette Score Plot')
    
    plt.xlabel('Silhouette Coefficient Values')
    plt.ylabel('Cluster Label')
    
    plt.savefig(filename)
    
    plt.close()


def main():
    data = load_data('politifact_factcheck_data.json')
    print(f'Loaded {len(data)} examples')
    
    labels = list(zip(*data))[1]
    label_counts = {label: labels.count(label) for label in set(labels)}
    pprint(label_counts)

    encoder = SentenceTransformer('bert-base-nli-mean-tokens')
    encoded_data = encode_data(encoder, data)

    train_data = split_data(encoded_data)
    model, metamodel = train_model(train_data, label_counts)

    X_train, X_test, y_train, y_test = train_data

    X_test_valid, y_test_valid = filter_data(model, metamodel, (X_test, y_test))
    y_pred = model.predict(X_test_valid)
    
    cm = confusion_matrix(y_test_valid, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(y_test_valid)))
    
    disp.plot()
    
    plt.savefig('confusion_matrix.png')
    
    plot_label_distribution(labels, 'label_distribution.png')

    sentences = list(zip(*data))[0]
    plot_sentence_length_distribution(sentences, 'sentence_length_distribution.png')

    embeddings, _ = zip(*encoded_data)
    embeddings = np.array(embeddings)

    plot_embedding_space(embeddings, labels, 'embedding_space_pca.png', method='pca')
    plot_embedding_space(embeddings, labels, 'embedding_space_tsne.png', method='tsne')

    plot_embedding_density(embeddings, 'embedding_density.png')
    plot_pairwise_distance_heatmap(embeddings, 'pairwise_distance_heatmap.png')
    plot_silhouette_score(embeddings, labels, 'silhouette_score.png')

    smote = SMOTE()

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    before_counts = Counter(y_train)
    after_counts = Counter(y_train_resampled)

    plot_class_balance(before_counts, after_counts, 'class_balance.png')


if __name__ == '__main__':
    main()
