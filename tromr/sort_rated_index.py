import sys

index_file = sys.argv[1]
with open(index_file, 'r') as f:
    lines = f.readlines()

lines.sort(key=lambda x: float(x.split(',')[2]))
for line in lines:
    print(line.strip())