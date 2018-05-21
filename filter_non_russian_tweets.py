import sys
import re

any_ru_cyrillic = re.compile('[\u0410-\u044f\u0401\u0451]')
nothing_but_ru_en = re.compile('[^0-9A-Za-z\u0410-\u044f\u0401\u0451 #\-\n]')

if len(sys.argv) < 2:
    print("Please specify file")

filename = sys.argv[1]
f_in = open(filename, 'r')

parts = filename.partition('.')
f_out = open(parts[0]+'_true-ru'+parts[1]+parts[2], 'w')

for line in f_in:
    match = nothing_but_ru_en.search(line)
    if not match and any_ru_cyrillic.search(line):
        f_out.write(line)
