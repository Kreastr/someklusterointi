import math
import os.path
import simplejson

def calculate_idfs(filename, force_recalc=False):
    if not force_recalc:
        if os.path.isfile(filename):
            with open(filename) as f:
                idfs_ascii = simplejson.load(f)

            idfs = {}
            for w, c in idfs_ascii.iteritems():
                idfs[w.encode('utf-8')] = c
             
            return idfs


    with open(filename) as f:
        lines = 0
        dfs = {}
        idfs = {}

        for line in f:
            lines += 1
            words_on_this_line = []
            if lines % 100000 == 0:
                print(str(lines))
            for word in line.strip().split(' ')[2:]:
                if word == '':
                    continue

                if word in words_on_this_line:
                    continue

                if word in dfs:
                    dfs[word] += 1
                else:
                    dfs[word] = 1

                words_on_this_line.append(word)

        for word, count in dfs.iteritems():
            idfs[word] = math.log(float(lines) / count)

        return idfs
