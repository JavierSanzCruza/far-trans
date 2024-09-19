#!/usr/bin/python
import os
import sys

__author__ = "Javier Sanz-Cruzado (javier.sanz-cruzadopuig@glasgow.ac.uk)"

if __name__ == "__main__":
    """
    Script for running basic dataset analysis.
    
    This program takes the FAR-Trans dataset and analyzes assets and customers over a set of 61 dates.
    
    Input:
    - Dataset path: the path on which the FAR-Trans dataset is stored. We assume names have not been changed.
    - Output directory: the path on which the results will be stored. A directory for every model will be created.
    
    Output:
    - For every tested date, two files will be created:
        - File 1: An asset file: assets_date.csv
                  Format: col_item \t current_price \t future_price \t ROI \t Annualized ROI \t Monthly ROI \t Volatility
        - File 2: An customer file: customers_date.csv
                  Format: customerID \t buy_price \t sell_price \t ROI \t Annualized ROI \t Monthly ROI
    - Summary files: four summary files. Format: timestamp \t stat1 \t stat2 \t ... \t statN
        - File 1: assets_1.csv: First 29 dates, asset summary
        - File 2: assets_2.csv: Last 31 dates, asset summary
        - File 3: customers_1.csv: First 29 dates, customer summary.
        - File 4: customers_2.csv: Last 31 dates, customer_summary.
        
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
    min_file = os.path.join(dataset_path, "limit_prices.csv")

    # Execute the algorithms:
    dates = [("2019-08-01", "2021-02-26", "28", "13", "6", output_directory, 1),
             ("2020-09-14", "2022-05-23", "31", "13", "6", output_directory, 2)]

    for date in dates:
        print("Starting analysis for period: " + str(date[0]) + " to " + str(date[1]))
        exec_code = "python ./dataset_analysis.py " + interactions_file + " " + time_series + " range " + \
                        date[0] + " " + date[1] + " " + date[2] + " " + date[3] + " " + date[5] + " " + date[4] \
                        + " assets_" + str(date[6]) + ".csv"
        if os.system(exec_code) != 0:
            sys.exit("Error when executing asset analysis for date" + str(date[0]) + " to " + str(date[1]))

        exec_code = "python ./customer_analysis.py " + interactions_file + " " + time_series + " " + min_file + " " \
                    + "range " + date[0] + " " + date[1] + " " + date[2] + " " + date[3] + " " + date[5] + " " \
                    + date[4] + " customers_" + str(date[6]) + ".csv"
        if os.system(exec_code) != 0:
            sys.exit("Error when executing customer analysis for date" + str(date[0]) + " to " + str(date[1]))
        print("End analysis for period: " + str(date[0]) + " to " + str(date[1]))

