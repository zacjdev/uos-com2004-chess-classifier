"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator
import collections as coll

N_DIMENSIONS = 10


def knn(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray, k):
    # K nearest neighbour
    k = 5

    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())

    if k == 1:
        # nearest neighbour
        nearest = np.argmax(dist, axis=1)  
        return train_labels[nearest]

    else: 
        x = np.dot(test, train.transpose())
        modtest = np.sqrt(np.sum(test * test, axis=1))
        modtrain = np.sqrt(np.sum(train * train, axis=1))
        dist = x / np.outer(modtest, modtrain.transpose())
        neighbours = np.argsort(-dist, axis=1)
        labels = []

        # Count the most common labels
        for char in range(dist.shape[0]):
            nearest_list = coll.Counter(train_labels[neighbours[char][:k]]).most_common(1)
            (nearest, _) = np.array(nearest_list)[0]
            labels.append(nearest)
    return np.array(labels)


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    print("train shape", train.shape)
    print("test shape", test.shape)
    print("train_labels shape", train_labels.shape)

    # n_images = test.shape[0]

    classified = knn(train, train_labels, test, 3)
    #classified = nn(train, train_labels, test)
    return classified

# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.

def makeA(data: np.ndarray):
    covx = np.cov(data, rowvar=0)
    n = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(n - 10, n - 1))
    v = np.fliplr(v)
    return v

def divergence(class1, class2):
    """compute a vector of 1-D divergences
    
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    
    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (
        1.0 / v1 + 1.0 / v2
    )
    return d12



def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    v = np.array(model['axes'])
    if v.size == 0:
        v = makeA(data)
        
        # only save eigenvector for final reduction of post-noise reduction PCA 
        model['axes'] = v.tolist()

    # PCA HERE
    # The images have all be pre-processed to be exactly 400 by 400 pixels, i.e., each square is 50 by 50 pixels
    
    # There are two label files: 
    # boards.train.json which provides labels for the training data, and 
    # boards.dev.json which provides labels for the test data.

    # compute first N pca axes

    # checking if training data eigenvalues have been calculated

    print(data.shape)

    pca_data = np.dot((data - np.mean(data)), v)

    print(pca_data.shape)

    return pca_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    model['classifier'] = 'knn' # 'nn' or 'knn'

    model["labels_train"] = labels_train.tolist()
    model['axes'] = np.array([]).tolist() 
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)
    
    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    print("board shape", fvectors_test.shape)
    return classify_squares(fvectors_test, model)
