import csv
with open('/home/apopiel/functions_copy.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        function = row['function']
        import os.path

        global___csv = '/lustre/scratch/apopiel/real/aoc/real_aoc_' + function + '_global_1.csv'
        isfile = os.path.exists(global___csv)
        print global___csv + ' ' + str(isfile)