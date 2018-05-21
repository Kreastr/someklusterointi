import glob

f_out = open("01.ru.txt", 'w')

for filename in glob.glob("*.ru.json"):
    print(filename)
    f = open(filename)

    for line in f:
        start_index = line.find('"text":"') + len('"text":"')
        i = start_index

        while  i < len(line):
            if line[i] == '"' and line [i-1] != '\\':
                break
            i += 1

        f_out.write(line[start_index:i].replace('\\n', ' ').replace('\\r', ' '))
        f_out.write('\n')

    f.close()

f_out.close()
