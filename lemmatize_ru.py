import sys
from pymystem3 import Mystem

if len(sys.argv) < 2:
    print("Please specify file")

filename = sys.argv[1]
f_in = open(filename, 'r')
if not f_in:
    exit(1)

parts = filename.partition('.')
f_out = open(parts[0]+'_lem'+parts[1]+parts[2], 'w')

if not f_out:
    f_in.close()
    exit(1)

m = Mystem()

line_count = 0
for line in f_in:
    line_count += 1
    if line_count % 1000 == 0:
        print(line_count)

    for lemma in m.lemmatize(line):
        f_out.write(lemma)
    f_out.write('\n')

f_out.close()
f_in.close()
