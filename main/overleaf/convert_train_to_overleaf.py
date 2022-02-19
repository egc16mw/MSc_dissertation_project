import csv
INPUT_FILE = "../output1/stats_overleaf.csv"
output = []
CHANGING_CONDITION = 5
with open(INPUT_FILE) as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        count = 0
        for line in row:
            if count >= len(output):
                output.append([])
            output[count].append(line)
            count += 1

epoch = []
accuracy = []
for item in output:
    if item[0] == "Epoch":
        epoch = item[1:]
    elif item[0] == "Accuracy":
        accuracy = item[1:]

if len(epoch) != len(accuracy):
    print("Length of epochs and accuracy don't match")
    print("Epoch: " + str(len(epoch)) + " and accuracy " + str(len(accuracy)))
    exit()

if len(epoch) < 0:
    print("Didn't find any epoch data")
    exit()

if len(accuracy) < 0:
    print("Didn't find any accuracy data")
    exit()


for i in range(0, int(len(epoch) / CHANGING_CONDITION) + 1):
    start_index = i * CHANGING_CONDITION
    end_index = i * CHANGING_CONDITION + CHANGING_CONDITION

    rows = []
    for e, a in zip(epoch[start_index:end_index], accuracy[start_index:end_index]):
        rows.append([e, a])

    file_name = "experiment1_" + str(i) + ".csv"
    with open(file_name, "w", newline="") as f:
        print("Writing file called: " + file_name)
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Accuracy"])
        for row in rows:
            writer.writerow(row)
