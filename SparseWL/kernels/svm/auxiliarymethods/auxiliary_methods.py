import math as m

import numpy as np
from scipy import sparse as sp
import sklearn.datasets as ds

# Read gram matrix in LIBSVM format.
def read_lib_svm(str):
    gm, c = ds.load_svmlight_file(str, multilabel=True)

    gram_matrix = gm.toarray()
    return gram_matrix[:, 1:], c


# Cosine normalization for a gram matrix.
def normalize_gram_matrix(gram_matrix):
    n = gram_matrix.shape[0]
    gram_matrix_norm = np.zeros([n, n], dtype=np.float64)

    for i in range(0, n):
        for j in range(i, n):
            if not (gram_matrix[i][i] == 0.0 or gram_matrix[j][j] == 0.0):
                g = gram_matrix[i][j] / m.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
                gram_matrix_norm[i][j] = g
                gram_matrix_norm[j][i] = g

    return gram_matrix_norm


# Cosine normalization for sparse feature vectors, i.e., \ell_2 normalization.
def normalize_feature_vector(feature_vectors):
    n = feature_vectors.shape[0]

    for i in range(0, n):
        norm = sp.linalg.norm(feature_vectors[i])
        feature_vectors[i] = feature_vectors[i] / norm

    return feature_vectors

