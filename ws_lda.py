import string
import sys
import nltk
from topic_model import Topic_model
nltk.download('punkt')
import logging

class WsLda:
    def __init__(self, queries, classes, labeled_entities):
        """
        :param queries: set of queries
        :param classes: set of predefined classes
        :param labeled_entities: dict containing entities and their possible classes
        """
        self.queries = queries
        self.classes = classes
        self.labeled_entities = labeled_entities

        # I_C
        self.class_index = {}
        # I_E
        self.named_entity_index = {}
        # T
        self.context_set = set()

        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(message)s")

    def preprocess_queries(self, queries):
        translator = str.maketrans('', '', string.punctuation)
        result = []
        for query in queries:
            # Remove puntuation
            query = query.translate(translator)
            result.append(' '.join(nltk.word_tokenize(query)))
        return result

    def train(self, max_iter_em=100, max_iter_inference=20, lambda_coef=0.1, epsilon_em=1e-4, epsilon_inference=1e-6):
        """
        :param max_iter_em: maximum number of iterations of the EM algorithm
        :param max_iter_inference: maximum number of iterations done in the inference in the E-step
        :param lambda_coef: Lambda coefficient. When set to 0 WS-LDA is the same as LDA
        :param epsilon_em: Used to determine convergence in the EM-algorithm
        :param epsilon_inference: Used to determine convergence in the inference in the E-step
        """
        logging.info("Started training WSLDA")
        entity2contexts = {}
        context2id = {}
        id2context = []
        for query in self.queries:
            for entity in self.labeled_entities.keys():
                if entity in query:
                    context = tuple([s.strip() for s in query.split(entity)])
                    contextID = context2id.get(context, -1)
                    if context not in id2context:
                        contextID = len(id2context)
                        context2id[context] = contextID
                        id2context.append(context)
                    if entity not in entity2contexts:
                        entity2contexts[entity] = [contextID]
                    else:
                        entity2contexts[entity].append(contextID)
        self.context_set = {id2context[contextID] for entity, contexts in entity2contexts.items() for contextID in contexts}
        logging.info("Extracted contexts for all entities, start training topic model")
        model = Topic_model(self.classes, self.labeled_entities, entity2contexts, context2id, id2context,
                            max_iter_em, max_iter_inference, lambda_coef, epsilon_em, epsilon_inference)
        beta, phi = model.train()
        logging.info("Topic model trained, start updating probability indices")
        # Update the class index with probabilities P(context|class)
        for i, topic in enumerate(self.classes):
            for j, probability in enumerate(beta[i]):
                self.class_index[(id2context[j]), topic] = probability
        # Update the named entity index with probabilities P(entity)
        for entity, contexts in entity2contexts.items():
            self.named_entity_index[entity] = len(contexts) / len(self.queries)
        # Update the named entity index with probabilities P(class|entity)
        for entity in phi:
            for classID, class_probabilities in enumerate(phi[entity].T):
                self.named_entity_index[(self.classes[classID], entity)] = sum(class_probabilities) / len(class_probabilities)
        logging.info("Indices updated, start rescanning the query log for acquired contexts")
        # Find new entities in the search query log using the contexts we found
        new_entity2contexts = {}
        for query in self.queries:
            for context in self.context_set:
                if context[0] in query and context[1] in query:
                    if query.index(context[0]) < query.index(context[1]) or not context[0] or not context[1]:
                        new_entity = self._get_entity(query, context).strip()
                        if new_entity not in entity2contexts:
                            if new_entity not in new_entity2contexts:
                                new_entity2contexts[new_entity] = [contextID]
                            else:
                                new_entity2contexts[new_entity].append(contextID)
        logging.info("Found " + str(len(new_entity2contexts)) + " new entities, start training topic model for these entities")
        rescan_model = Topic_model(self.classes, {}, new_entity2contexts, context2id, id2context,
                            max_iter_em, max_iter_inference, lambda_coef, epsilon_em, epsilon_inference)
        rescan_beta, rescan_phi = rescan_model.train(beta)
        logging.info("Rescand topic model trained, start updating indices")
        # Update the named entity index with new entities with probabilities P(entity)
        for entity, contexts in new_entity2contexts.items():
            self.named_entity_index[entity] = len(contexts) / len(self.queries)
        # Update the named entity index with new entities with probabilities P(class|entity)
        for entity in rescan_phi:
            for classID, class_probabilities in enumerate(rescan_phi[entity].T):
                self.named_entity_index[(self.classes[classID], entity)] = sum(class_probabilities) / len(class_probabilities)
        logging.info("Indices updated")



    def _get_entity(self, query, context):
        """
        :param query: The query to be searched 
        :param context: query - entity
        :return: all words in the query between the left side of the context and the right side
        """
        if not context[0]:
            return query.split(context[1])[0]
        elif not context[1]:
            return query.split(context[0])[1]
        else:
            return query.split(context[0])[1].split(context[1])[0]

    def get_best_entities(self, query):
        best_entities = []
        query = self.preprocess_queries([query])[0]
        words = query.split()
        for i in range(len(words)):
            for j in range(i, len(words)):
                entity = ' '.join(words[i:j+1]).strip()
                context = (' '.join(words[:i]).strip(), ' '.join(words[j+1:]).strip())
                if entity in self.named_entity_index and context in self.context_set:
                    for c in self.classes:
                        probability = self.named_entity_index[entity] * self.named_entity_index[(c, entity)] * self.class_index[(context, c)]
                        best_entities.append({"entity": entity, "class": c, "probability": probability})
        return sorted(best_entities, key=lambda k: k['probability'])

    def get_best_entity(self, query):
        best_entity = self.get_best_entities(query)[0]
        return best_entity['entity'], best_entity['class'], best_entity['probability']
