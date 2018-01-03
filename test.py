import datetime
from random import randint
import sys

import itertools
from ws_lda import WsLda
import pickle
import pathlib

import string
import nltk
nltk.download('punkt')
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(message)s")


def preprocess_queries(queries):
    translator = str.maketrans(' ', ' ', string.punctuation)
    result = []
    for query in queries:
        # Remove puntuation
        query = query.translate(translator)
        result.append(' '.join(nltk.word_tokenize(query)))
    return result


def train(input_queries, input_labels, output_dir, remove_test_data=False):
    with open(input_queries) as f:
        queries = [q.strip() for q in f.readlines() if len(q) < 100]
    with open(input_labels) as f:
        labels = {}
        for line in f:
            labels[line.split(':')[0]] = {c.strip() for c in line.split(':')[1].split(',')}
    classes = [c for c in {c for l in labels.keys() for c in labels[l]}]

    if remove_test_data:
        num_test_queries = 5000
        test_query_indices = {randint(0, len(queries)) for i in range(num_test_queries)}
        test_queries = {queries[i] for i in test_query_indices}
        for test_query in sorted(test_query_indices, reverse=True):
            queries.pop(test_query)
        current_time = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
        with open("test_data/aol_data/queries" + current_time + ".txt", 'w') as test_file:
            test_file.write("\n".join(test_queries))
        with open("training_data/aol_data/queries_minus_5000_test.txt", 'w') as test_file:
            test_file.write("\n".join(queries))
    queries = preprocess_queries(queries)
    print(queries)
    model = WsLda(queries, classes, labels)
    model.train()
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_dir + "/class_index.pickle", 'wb') as f:
        pickle.dump(model.class_index, f)
    with open(output_dir + "/named_entity_index.pickle", 'wb') as f:
        pickle.dump(model.named_entity_index, f)
    with open("backup/" + output_dir + "/class_index.pickle", 'wb') as f:
        pickle.dump(model.class_index, f)
    with open("backup/" + output_dir + "/named_entity_index.pickle", 'wb') as f:
        pickle.dump(model.named_entity_index, f)
    print(model.class_index)
    print(model.named_entity_index)

train("training_data/toy_data/queries.txt", "training_data/toy_data/labels.txt", "indices/toy_data")
#train("training_data/twitter_data/all_tweets.txt", "training_data/twitter_data/all_labels.txt", "indices/twitter_data")
#train("training_data/aol_data/queries.txt", "training_data/aol_data/labels.txt", "indices/aol_data", True)

#with open("indices/aol_data/rescanned_entities.pickle", 'rb') as f:
#    rescanned_entities = pickle.load(f)
#    with open("indices/aol_data/rescanned_entities.txt") as plain_f:
#        plain_f.write(rescanned_entities)



def preprocess_aol():
    with open("../../data/aol-data/AOL-user-ct-collection/user-ct-test-collection-01.txt") as in_file:
        with open("training_data/aol_data/queries.txt", 'w') as out_file:
            previous_query = ""
            for line in in_file:
                query = line.split("\t")[1]
                if previous_query != query and query != '-':
                    out_file.write(query + "\n")
                previous_query = query

#preprocess_aol()

def preprocess_tweets():
    with open("training_data/twitter_data/original_data.txt") as f:
        with open("training_data/twitter_data/all_tweets.txt", 'w') as all_tweets:
            with open("training_data/twitter_data/all_labels.txt", 'w') as all_labels:
                tweet = ""
                entity = ""
                entity_type = ""
                labels = {}
                for line in f:
                    if len(line.split()) < 2:
                        all_tweets.write(tweet.strip() + "\n")
                        tweet = ""
                    else:
                        word = line.split()[0]
                        tweet += " " + word
                        topic = line.split()[1]
                        if topic == "O" and entity:
                            if entity not in labels:
                                labels[entity] = {entity_type}
                            else:
                                labels[entity].add(entity_type)
                            entity = ""
                        if topic.startswith("B-"):
                            entity = word
                            entity_type = topic.split("B-")[1]
                        elif topic.startswith("I-"):
                            entity += " " + word
                for entity, topics in labels.items():
                    all_labels.write(entity.strip() + ":" + ",".join(topics) + "\n")
