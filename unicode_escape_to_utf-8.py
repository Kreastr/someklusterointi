import sys

if len(sys.argv) < 2:
    print("Please specify file")

filename = sys.argv[1]
f_in = open(filename, 'r')
if not f_in:
    exit(1)

parts = filename.partition('.')
f_out = open(parts[0]+'_utf-8'+parts[1]+parts[2], 'w')

if not f_out:
    f_in.close()
    exit(1)

for line in f_in:
    f_out.write(line.decode('unicode-escape').encode('utf-8'))

f_out.close()
f_in.close()
