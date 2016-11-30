# AI HW3 Part 2

from math import log
from collections import defaultdict
import numpy as np



# Multinomial Naive Bayes Model Classifier
def multinomial(test_file, unique_words, pos_words, neg_words, k, pos_prior, neg_prior):
    ''' Feature likelihood = conditional probability for each word, P(word|class)
        Likelihood = sum of logs of feature likelihoods, P(document|class)
        Posterior = sum of logs of prior and likelihoods, P(class) * P(document|class)
        Prior = documents in class / total number of documents, P(class)
    '''
    # Number of words with positive and negative labels
    n_pos_words = 0
    n_neg_words = 0
    # Conditional probability (feature likelihood) tables
    pos_table = {}
    neg_table = {}

    # Find number of words with positive and negative labels
    for word in pos_words:
        n_pos_words += pos_words[word]
    for word in neg_words:
        n_neg_words += neg_words[word]

    # Build conditional probability (feature likelihood) tables
    for word in unique_words:
        pos_table[word] = (pos_words[word] + k) / (n_pos_words + k*len(pos_words))
        neg_table[word] = (neg_words[word] + k) / (n_neg_words + k*len(neg_words))

    ################################# Testing #################################

    # Parse testing file, build document dictionary, and classify documents
    with open(test_file, 'r') as f:
        n_correct = 0
        n_incorrect = 0
        confusion_matrix = np.zeros((2,2))

        # Loop through documents
        for line in f:
            words = line.split()
            label = int(words.pop(0))

            # Build dictionary
            doc_words = defaultdict(int)
            for string in words:
                pair = string.split(':')
                doc_words[pair[0]] += int(pair[1])

            # Calculate likelihoods for each class
            pos_likelihood = 0
            neg_likelihood = 0

            for word in doc_words:
                count = doc_words[word]
                # Skip words that are not in dictionary
                if word in unique_words:
                    # Repeat words if needed
                    for i in range(count):
                        pos_likelihood += log(pos_table[word])
                        neg_likelihood += log(neg_table[word])

            # Calculate posteriors for each class
            pos_posterior = log(pos_prior) + pos_likelihood
            neg_posterior = log(neg_prior) + neg_likelihood

            if pos_posterior >= neg_posterior:
                prediction = 1
            elif pos_posterior < neg_posterior:
                prediction = -1

            if label == prediction:
                n_correct += 1
                if label == 1:
                    confusion_matrix[0][0] += 1
                elif label == -1:
                    confusion_matrix[1][1] += 1
            else:
                n_incorrect += 1
                if label == 1:
                    confusion_matrix[0][1] += 1
                elif label == -1:
                    confusion_matrix[1][0] += 1



    # Normalize confusion matrix
    pos_sum = confusion_matrix[0][0] + confusion_matrix[0][1]
    neg_sum = confusion_matrix[1][0] + confusion_matrix[1][1]
    confusion_matrix[0][0] = confusion_matrix[0][0] / pos_sum
    confusion_matrix[0][1] = confusion_matrix[0][1] / pos_sum
    confusion_matrix[1][0] = confusion_matrix[1][0] / neg_sum
    confusion_matrix[1][1] = confusion_matrix[1][1] / neg_sum

    # Find top 10 words with highest likelihood for each class
    sorted_pos = sorted(pos_table, key=lambda x: pos_table[x])
    sorted_neg = sorted(neg_table, key=lambda x: neg_table[x])

    top_pos_like = sorted_pos[0:10]
    top_neg_like = sorted_neg[0:10]

    # Find top 10 words with highest odds ratio for each class
    pos_odds = {}
    neg_odds = {}
    for word in unique_words:
        pos_odds[word] = -(log(pos_table[word]) - log(neg_table[word]))
        neg_odds[word] = -(log(neg_table[word]) - log(pos_table[word]))

    sorted_pos_odds = sorted(pos_odds, key=lambda x: pos_odds[x])
    sorted_neg_odds = sorted(neg_odds, key=lambda x: neg_odds[x])
    top_pos_odds = sorted_pos_odds[0:10]
    top_neg_odds = sorted_neg_odds[0:10]

    return (n_correct, n_incorrect, confusion_matrix, top_pos_like, top_neg_like,
            top_pos_odds, top_neg_odds)



# Bernoulli Naive Bayes Model Classifier
def bernoulli(test_file, unique_words, pos_docs, neg_docs, k, pos_prior, neg_prior,
                    n_pos_docs, n_neg_docs):
    ''' Feature likelihood = conditional probability for each word, P(word|class)
        Likelihood = sum of logs of feature likelihoods, P(document|class)
        Posterior = sum of logs of prior and likelihoods, P(class) * P(document|class)
        Prior = documents in class / total number of documents, P(class)
    '''
    # Conditional probability (feature likelihood) tables
    pos_table = {}
    neg_table = {}

    # Build conditional probability (feature likelihood) tables
    for word in unique_words:
        pos_table[word] = (pos_docs[word] + k) / (n_pos_docs + k*len(pos_docs))
        neg_table[word] = (neg_docs[word] + k) / (n_neg_docs + k*len(neg_docs))

    ################################# Testing #################################

    # Parse testing file, build dictionary, and classify test cases
    with open(test_file, 'r') as f:
        n_correct = 0
        n_incorrect = 0
        confusion_matrix = np.zeros((2,2))

        # Loop through documents
        for line in f:
            words = line.split()
            label = int(words.pop(0))

            # Build dictionary
            doc_words = defaultdict(int)
            for string in words:
                pair = string.split(':')
                doc_words[pair[0]] += int(pair[1])

            # Calculate likelihoods for each class
            pos_likelihood = 0
            neg_likelihood = 0

            for word in doc_words:
                count = doc_words[word]
                # Skip words that are not in dictionary
                if word in unique_words:
                    # Repeat words if needed
                    for i in range(count):
                        pos_likelihood += log(pos_table[word])
                        neg_likelihood += log(neg_table[word])

            pos_posterior = log(pos_prior) + pos_likelihood
            neg_posterior = log(neg_prior) + neg_likelihood

            if pos_posterior >= neg_posterior:
                prediction = 1
            elif pos_posterior < neg_posterior:
                prediction = -1

            if label == prediction:
                n_correct += 1
                if label == 1:
                    confusion_matrix[0][0] += 1
                elif label == -1:
                    confusion_matrix[1][1] += 1
            else:
                n_incorrect += 1
                if label == 1:
                    confusion_matrix[0][1] += 1
                elif label == -1:
                    confusion_matrix[1][0] += 1

    # Normalize confusion matrix
    pos_sum = confusion_matrix[0][0] + confusion_matrix[0][1]
    neg_sum = confusion_matrix[1][0] + confusion_matrix[1][1]
    confusion_matrix[0][0] = confusion_matrix[0][0] / pos_sum
    confusion_matrix[0][1] = confusion_matrix[0][1] / pos_sum
    confusion_matrix[1][0] = confusion_matrix[1][0] / neg_sum
    confusion_matrix[1][1] = confusion_matrix[1][1] / neg_sum

    # Find top 10 words with highest likelihood for each class
    sorted_pos = sorted(pos_table, key=pos_table.__getitem__)
    sorted_neg = sorted(neg_table, key=neg_table.__getitem__)

    top_pos_like = sorted_pos[0:10]
    top_neg_like = sorted_neg[0:10]

    # Find top 10 words with highest odds ratio for each class
    pos_odds = {}
    neg_odds = {}
    for word in unique_words:
        pos_odds[word] = -(log(pos_table[word]) - log(neg_table[word]))
        neg_odds[word] = -(log(neg_table[word]) - log(pos_table[word]))

    sorted_pos_odds = sorted(pos_odds, key=lambda x: pos_odds[x])
    sorted_neg_odds = sorted(neg_odds, key=lambda x: neg_odds[x])
    top_pos_odds = sorted_pos_odds[0:10]
    top_neg_odds = sorted_neg_odds[0:10]

    return (n_correct, n_incorrect, confusion_matrix, top_pos_like, top_neg_like,
            top_pos_odds, top_neg_odds)



# Main function
def main():
    ''' Main function '''
    # Select which Naive Bayes Model to use
    model = input('\nSelect Naive Bayes model:\n'
                '1: Multinomial Naive Bayes\n'
                '2: Bernoulli Naive Bayes\n'
                'Your choice: ')

    # Select which input corpus to use
    corpus = input('\nSelect corpus:\n'
                '1: Sentiment analysis of movie reviews\n'
                '2: Binary conversation topic classifiers\n'
                'Your choice: ')

    if corpus == '1':
        train_file = 'movie_review/rt-train.txt'
        test_file = 'movie_review/rt-test.txt'
    elif corpus == '2':
        train_file = 'fisher_2topic/fisher_train_2topic.txt'
        test_file = 'fisher_2topic/fisher_test_2topic.txt'
    else:
        print('Error')
        return

    ################################ Training #################################

    # Dictionaries for word counts for unique words, positive and negative labels,
    # and document counts of positive and negative labels
    unique_words = defaultdict(int)
    pos_words = defaultdict(int)
    neg_words = defaultdict(int)
    pos_docs = defaultdict(int)
    neg_docs = defaultdict(int)
    # Number of documents with positive and negative labels
    n_pos_docs = 0
    n_neg_docs = 0

    # Parse training file and build dictionaries
    with open(train_file, 'r') as f:
        # Loop through documents
        for line in f:
            words = line.split()
            label = int(words.pop(0))

            # Accumulate number of positive and negative documents
            if label == 1:
                n_pos_docs += 1
            elif label == -1:
                n_neg_docs += 1

            # Build dictionaries
            for string in words:
                pair = string.split(':')
                unique_words[pair[0]] += int(pair[1])
                if label == 1:
                    pos_words[pair[0]] += int(pair[1])
                    pos_docs[pair[0]] += 1
                elif label == -1:
                    neg_words[pair[0]] += int(pair[1])
                    neg_docs[pair[0]] += 1

    # Calculate priors for each class
    pos_prior = n_pos_docs / (n_pos_docs + n_neg_docs)
    neg_prior = n_neg_docs / (n_pos_docs + n_neg_docs)

    # K values
    k_values = [0.1, 0.2, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

    # Test for each k value
    for k in k_values:
        print('\nResults for k =', k)

        # Perform classification based on selected model
        if model == '1':
            retval = multinomial(test_file, unique_words, pos_words, neg_words, k,
                                    pos_prior, neg_prior)
        elif model == '2':
            retval = bernoulli(test_file, unique_words, pos_docs, neg_docs, k,
                                    pos_prior, neg_prior, n_pos_docs, n_neg_docs)
        else:
            print('Error')
            return

        accuracy = retval[0] / (retval[0] + retval[1])
        #print('Correct:', retval[0], 'Incorrect:', retval[1])
        print('Accuracy:', accuracy)

    # Enter Laplacian smoothing constant
    k = float(input('\nEnter Laplacian smoothing constant: '))

    print('\n\n################ Results for k =', k, '################')

    # Perform classification based on selected model
    if model == '1':
        retval = multinomial(test_file, unique_words, pos_words, neg_words, k,
                                    pos_prior, neg_prior)
    elif model == '2':
        retval = bernoulli(test_file, unique_words, pos_docs, neg_docs, k,
                                    pos_prior, neg_prior, n_pos_docs, n_neg_docs)
    else:
        print('Error')
        return

    accuracy = retval[0] / (retval[0] + retval[1])

    print('\nPrior for positive label:', pos_prior)
    print('Prior for negative label:', neg_prior)

    print('\nCorrect:', retval[0], 'Incorrect:', retval[1])

    print('\nAccuracy:', accuracy)

    print('\nConfusion Matrix:\n', retval[2])

    print('\nTop 10 Likelihoods:\n\nPositive:')
    for word in retval[3]:
        print(word)
    print('\nNegative:')
    for word in retval[4]:
        print(word)

    print('\nTop 10 Odds Ratios:\n\nPositive:')
    for word in retval[5]:
        print(word)
    print('\nNegative:')
    for word in retval[6]:
        print(word)



# Main function call
if __name__ == '__main__':
    main()