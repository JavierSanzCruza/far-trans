#!/usr/bin/python
import os
import sys

__author__ = "Javier Sanz-Cruzado (javier.sanz-cruzadopuig@glasgow.ac.uk)"

if __name__ == "__main__":
    """
    Script for executing the experiments in the FAR-Trans paper, except for the hybrid models.
    For the experiments with hybrid models, please, see run_hybrid_experiments_script.py.
    
    This program takes the FAR-Trans dataset and runs a series of recommender systems over
    the data (basic models, profitability prediction and collaborative filtering methods).
    Then, it evaluates them over a set of 61 dates.
    
    Input:
    - Dataset path: the path on which the FAR-Trans dataset is stored. We assume names have not been changed.
    - Output directory: the path on which the results will be stored. A directory for every model will be created.
    
    Output:
    - A directory for every recommendation algorithm.
    - For every tested date, three files will be created:
        - File 1: A recommendation file. titled "algorithm_date_recs.csv".
                  Format: col_user \t col_item \t col_rating (sorted by user in ascending order, rating in descending)
        - File 2: An evaluation file, titled "algorithm_date_metrics.csv"
                  Format: metric \t value
        - File 3: A customer evaluation file, where the metrics for every customer are detailed:
                  Format: col_user \t metric1 \t metric2 ... \t metricN
    """

    if len(sys.argv) < 3:
        sys.stderr.write("ERROR: Invalid arguments")
        sys.stderr.write("\tdataset_path: route to the dataset.")
        sys.stderr.write("\toutput_dir: directory on which to store the results.")

    dataset_path = sys.argv[1]
    output_directory = sys.argv[2]

    # Obtain the routes for the interaction and time series files.
    interactions_file = os.path.join(dataset_path, "transactions.csv")
    time_series = os.path.join(dataset_path, "close_prices.csv")

    # Execute the algorithms:
    dates = [("2019-08-01", "2021-02-26", "28", "13", "6", output_directory),
             ("2020-09-14", "2022-05-23", "31", "13", "6", output_directory)]

    algorithms = [("random", "random"),
                  ("rfr", "rfr", "20", "full_short"),
                  ("lr", "lr", "full_short"),
                  ("lgbm", "lgbm", "full_short"),
                  ("pop", "pop"),
                  ("arm", "arm"),
                  ("ubknn", "ubknn", "cosine", "16", "False"),
                  ("mf", "mf", "128", "bpr", "mf/backup"),
                  ("lightgcn", "lightgcn", "64", "0.01", "lightgcn/backup")]

    hybrid_algs = [("lambdamart", "hybrid-ndcg", "ndcg"),
                   ("lambdamart", "hybrid-regression", "regression")]

    for date in dates:
        for hybrid in hybrid_algs:
            print("Starting hybrid-" + hybrid[2] + " for a time horizon of " + date[4] + " month(s)")

            directory = os.path.join(date[5], hybrid[1])
            if not os.path.exists(directory):
                os.makedirs(directory)

            exec_code = "python ./recommendation.py " + interactions_file + " " + time_series + " range " + \
                        date[0] + " " + date[1] + " " + date[2] + " " + date[3] + " " + directory + " " + date[4] \
                        + " None "

            for i in range(2, len(hybrid)):
                exec_code += " " + hybrid[i]
            for algorithm in algorithms:
                exec_code += " " + algorithm[0] + os.path.join(date[5], algorithm[1]) + " " + str(len(algorithm) - 2)
                if algorithm[1].__contains__("cf"):
                    for i in range(2, len(algorithm) - 1):
                        exec_code += " " + algorithm[i]
                    directory2 = os.path.join(date[5], algorithm[-1])
                    if not os.path.exists(directory2):
                        os.makedirs(directory2)
                    exec_code += " " + directory2
                else:
                    for i in range(2, len(algorithm)):
                        exec_code += " " + algorithm[i]

            if os.system(exec_code) != 0:
                sys.exit("Error when executing algorithm " + str(hybrid) + " for date " + str(date))

            print("Ending algorithm hybrid-" + hybrid[2] + " for " + date[4] + " month(s)")
