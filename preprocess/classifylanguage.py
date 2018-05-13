import csv
import sys
from langdetect import detect
from collections import defaultdict

dataset_name = sys.argv[1]

# Keep record of languages counting
languages_counter = defaultdict(int)

with open(dataset_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        try:
            language = detect(row['TextPost'])
            languages_counter[language] += 1
        except:
            languages_counter['unknown'] += 1

with open('output/language_count.csv', 'w') as count_file:
    count_file.write('language;count\n')
    for k, v in languages_counter.items():
        count_file.write(f'{k};{v}\n')
