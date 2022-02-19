import csv
import numpy as np
INPUT_FILE = "../../Experiment 3/results/alpha.csv"
OUTPUT_FILE = "../../Experiment 3/alpha.dat"
output = []
print(np.load("../../Experiment 3/results/intents3.npy", allow_pickle=True).item())
with open(INPUT_FILE) as f:
    csv_reader = csv.reader(f, delimiter=',')
    row_count = 0
    for row in csv_reader:
        column_count = 1
        if row_count == 0:
            pass
        else:
            temp_output = []
            for value in row[1:]:
                temp_temp_output = []
                temp_temp_output.append(column_count)
                temp_temp_output.append(row_count)
                temp_temp_output.append(value)
                temp_output.append(temp_temp_output)
                column_count += 1
            output.append(temp_output)
        row_count += 1

with open(OUTPUT_FILE, "w") as f:
    for x in output:
        for line in x:
            temp = str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + "\n"
            f.write(temp)
        f.write("\n")