import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN0, GINE0, GIN, GINE


def main():
    num_reps = 10

    # Smaller datasets.
    dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["NCI109", True], ["PROTEINS", True],
               ["PTC_FM", True], ["REDDIT-BINARY", False], ["ENZYMES", True]]

    results = []
    for d, use_labels in dataset:
        dp.get_dataset(d)

        acc, s_1, s_2 = gnn_evaluation(GIN0, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True)
        print(d + " " + "GIN0 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN0 " + str(acc) + " " + str(s_1) + " " + str(s_2))

        acc, s_1, s_2 = gnn_evaluation(GIN, d, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True)
        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))

    num_reps = 3

    # Larger datasets with edge labels.
    dataset = [["Yeast", True], ["YeastH", True], ["UACC257", True], ["UACC257H", True], ["OVCAR-8", True],
               ["OVCAR-8H", True]]
    dataset = [["YeastH", True], ["UACC257", True], ["UACC257H", True], ["OVCAR-8", True],
               ["OVCAR-8H", True]]

    for d, use_labels in dataset:
        dp.get_dataset(d)

        acc, s_1, s_2 = gnn_evaluation(GINE, d, [2], [64], max_num_epochs=200,
                                       batch_size=64, start_lr=0.01,
                                       num_repetitions=num_reps, all_std=True)
        print(d + " " + "GINE " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINE " + str(acc) + " " + str(s_1) + " " + str(s_2))

        acc, s_1, s_2 = gnn_evaluation(GINE0, d, [2], [64], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01,
                                       num_repetitions=num_reps, all_std=True)
        print(d + " " + "GINE0 " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GINE0 " + str(acc) + " " + str(s_1) + " " + str(s_2))


if __name__ == "__main__":
    main()
