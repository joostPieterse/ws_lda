import math
import numpy
import logging
import sys
from scipy.special import psi, digamma, polygamma


class Topic_model:
    def __init__(self, classes, labeled_entities, entity2contexts, context2id, id2context, max_iter_em,
                 max_iter_inference, lambda_coef, epsilon_em, epsilon_inference):
        """
        :param classes: set of predefined classes
        :param labeled_entities: dict containing entities and their possible classes
        :param entity2contexts: 
        :param context2id: 
        :param id2context: 
        :param max_iter_em: maximum number of iterations of the EM algorithm
        :param max_iter_inference: maximum number of iterations done in the inference in the E-step
        :param lambda_coef: Lambda coefficient. When set to 0 WS-LDA is the same as LDA
        :param epsilon_em: Used to determine convergence in the EM-algorithm
        :param epsilon_inference: Used to determine convergence in the inference in the E-step
        """
        self.classes = classes
        self.labeled_entities = labeled_entities
        self.max_iter_em = max_iter_em
        self.max_iter_inference = max_iter_inference
        self.lambda_coef = lambda_coef
        self.epsilon_em = epsilon_em
        self.epsilon_inference = epsilon_inference

        self.entity2contexts = entity2contexts
        self.context2id = context2id
        self.id2context = id2context

        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(message)s")


    def train(self, beta=None):
        """
        Offline training for WS-LDA        
        :param beta: use this fixed beta during training instead of updating it during the EM algorithm
        :return: beta, phi
        """
        # Number of topics
        K = len(self.classes)
        # Number of documents
        M = len(self.entity2contexts.keys())
        # Nd
        word_count_per_doc = {entity: len(contexts) for entity, contexts in self.entity2contexts.items()}
        max_Nd = max(word_count_per_doc.values())
        # Total number of words
        N = sum(word_count_per_doc.values())
        # Total number of unique words (i.e. vocabulary size)
        V = len({word for words in self.entity2contexts.values() for word in words})

        logging.info("%s classes", K)
        logging.info("%s entities (=documents)", M)
        logging.info("%s distinct contexts", V)

        # Indicator variable for each (entity, class)
        y = {entity: [0 for i in range(K)] for entity in self.entity2contexts.keys()}
        for entity, labels in self.labeled_entities.items():
            for label in labels:
                y[entity][self.classes.index(label)] = 1
        alpha = 50 / K
        update_beta = False
        if beta is None:
            beta = numpy.zeros([K, V]) + 1 / V
            update_beta = True
        phi = {}
        gamma = {}
        for d in self.entity2contexts.keys():
            phi[d] = numpy.zeros([len(self.entity2contexts[d]), K]) + 1 / K
            gamma[d] = [alpha + N / K for i in range(K)]

        logging.info("Starting EM-algorithm")
        converged_em = False
        num_iterations_em = 0
        while not converged_em and num_iterations_em < self.max_iter_em:
            logging.info("Iteration %s/%s of EM-algorithm", num_iterations_em, self.max_iter_em)
            # E-step
            for doc_index, d in enumerate(self.entity2contexts.keys()):
                converged_inference = False
                num_iterations_inference = 0
                while not converged_inference and num_iterations_inference < self.max_iter_inference:
                    num_iterations_inference += 1
                    sum_gamma_d = sum(gamma[d])
                    for i in range(K):
                        rhs = math.exp(psi(gamma[d][i]) - psi(sum_gamma_d) + self.lambda_coef / N * y[d][i])
                        for n, v in enumerate(self.entity2contexts[d]):
                            phi[d][n][i] = beta[i][v] * rhs
                    # Normalize phi
                    for n, v in enumerate(self.entity2contexts[d]):
                        phi[d][n] /= sum(phi[d][n])
                    for i in range(K):
                        gamma[d][i] = alpha + sum(phi[d].T[i])

                    """
                    for n, v in enumerate(self.entity2contexts[d]):
                        for i in range(K):
                            phi[d][n][i] = beta[i][v] * math.exp(
                                psi(gamma[d][i]) - psi(sum_gamma_d) + self.lambda_coef / N * y[d][i]
                            )
                        # Normalize phi
                        phi[d][n] /= sum(phi[d][n])
                    gamma[d] = alpha + sum(phi[d])
                    """
            # M-step
            # Update beta
            logging.info("Started updating beta")
            if update_beta:
                beta = numpy.zeros([K, V])
                for d in self.entity2contexts.keys():
                    for n, v in enumerate(self.entity2contexts[d]):
                        for i in range(K):
                            # Beta_ij = the sum of all phi_dni for all words w_dn that are equal to word j
                            beta[i][v] += phi[d][n][i]
                for i in range(K):
                    beta[i] /= sum(beta[i])
            logging.info("Beta updated, start updating alpha")
            # Update alpha using Newton-Rhapson
            log_alpha = math.log(alpha)
            dL = M * (K * digamma(K * alpha) - K * digamma(alpha) + sum(psi(gamma[d][i] - psi(sum(gamma[d][j] for j in range(K)))) for d in self.entity2contexts.keys()))
            ddL = M * (K ** 2 * polygamma(1, K * alpha) - K * polygamma(1, alpha))
            try:
                alpha = math.exp(log_alpha - dL / (ddL * alpha + dL))
            except OverflowError:
                # Don't update I guess
                pass
            alpha = max(alpha, 1e-50)
            num_iterations_em += 1
        return beta, phi
