import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def read_classes(ds_name):
    with open("../datasets/" + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return classes


def main():
    path = "./svm/SVM/src/EXPSPARSE/"

    for name in ["Yeast", "YeastH", "UACC257", "UACC257H", "OVCAR-8", "OVCAR-8H"]:
        for algorithm in ["WL", "LWL2", "LWLP2"]:

            print(name)
            print(algorithm)

            # Collect feature matrices over all iterations
            all_feature_matrices = []
            classes = read_classes(name)
            classes = np.array(classes)
            for i in range(0, 6):
                # Load feature matrices.
                feature_vector = pd.read_csv(path + name + "__" + algorithm + "_" + str(i), header=1,
                                             delimiter=" ").to_numpy()

                feature_vector = feature_vector.astype(int)
                feature_vector[:, 0] = feature_vector[:, 0] - 1
                feature_vector[:, 1] = feature_vector[:, 1] - 1
                feature_vector[:, 2] = feature_vector[:, 2] + 1

                xmax = int(feature_vector[:, 0].max())
                ymax = int(feature_vector[:, 1].max())

                feature_vector = sp.coo_matrix((feature_vector[:, 2], (feature_vector[:, 0], feature_vector[:, 1])),
                                               shape=(xmax + 1, ymax + 1))
                feature_vector = feature_vector.tocsr()

                all_feature_matrices.append(feature_vector)
            print("### Data loading done.")

            test_accuracies_all = []
            for _ in range(10):

                kf = KFold(n_splits=10, shuffle=True)
                test_accuracies = []
                for train_index, test_index in kf.split(list(range(len(classes)))):
                    best_f = None
                    best_m = None
                    best_val = 0.0
                    for f in all_feature_matrices:
                        train_index, val_index = train_test_split(train_index, test_size=0.1)
                        train = f[train_index]
                        val = f[val_index]
                        c_train = classes[train_index]
                        c_val = classes[val_index]
                        for c in [10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]:
                            clf = LinearSVC(C=c)
                            clf.fit(train, c_train)
                            p = clf.predict(val)
                            a = np.sum(np.equal(p, c_val)) / val.shape[0]

                            if a > best_val:
                                best_val = a
                                best_f = f
                                best_m = clf

                    test = best_f[test_index]
                    c_test = classes[test_index]
                    p = best_m.predict(test)
                    a = np.sum(np.equal(p, c_test)) / test.shape[0]
                    test_accuracies.append(a * 100.0)

                test_accuracies_all.append(np.mean(test_accuracies))

            print(np.mean(test_accuracies_all), np.std(test_accuracies_all))

if __name__ == "__main__":
    main()
