import csv

def read_lines():
    with open('wavelet.csv', 'rU') as data:
        reader = csv.reader(data)
        for row in reader:
            yield [ float(i) for i in row ]

for i in read_lines():
    print(i)

# to get a list, instead of a generator, use
xy = list(read_lines())