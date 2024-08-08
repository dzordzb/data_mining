import logging
import json
from pprint import pprint
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

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

def train_model(data, test_size=0.4):
    logging.info('Training model...')
    
    X_train, X_test, y_train, y_test = data
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

def main():
    data = load_data('politifact_factcheck_data.json')
    print(f'Loaded {len(data)} examples')
    
    labels = list(zip(*data))[1]
    label_counts = {label: labels.count(label) for label in set(labels)}
    pprint(label_counts)

    encoder = SentenceTransformer('bert-base-nli-mean-tokens')
    encoded_data = encode_data(encoder, data)

    X_train, X_test, y_train, y_test = split_data(encoded_data)

    print(f'Training set size: {len(X_train)}')
    print(f'Test set size: {len(X_test)}')

    model, metamodel = train_model((X_train, X_test, y_train, y_test))
    print("Model and metamodel trained successfully")

if __name__ == '__main__':
    main()