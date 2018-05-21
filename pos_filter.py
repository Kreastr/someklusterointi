from pymystem3 import Mystem
from itertools import takewhile

mystem = Mystem()

# from /home/aminoff/opencorp/corpanalyzer/w2v_classifier.py
def pos_filter(strng, poslist):
    if not strng:
        return [] if outtype is list else u''
    if type(strng) not in [unicode, str]:
        anstr = " ".join(filter(lambda l: l != " ", strng))
    else:
        anstr = strng
    analysis = filter(lambda tk: all(map(lambda c: c.isalnum() or c == u'-', tk[u'text'])), mystem.analyze(anstr))

    if '-' in anstr:
        i = 0
        while 1:
            #try:
            if i >= len(analysis):
                break
            if analysis[i][u'text'] == u'-':
                if i == len(analysis)-1:
                    del analysis[i]
                    break
                elif i == 0:
                    del analysis[i]
                    continue
                newtoken = analysis[i+1]
                if newtoken[u'analysis'] if u'analysis' in newtoken else False:
                    if analysis[i-1][u'analysis'] if u'analysis' in analysis[i - 1] else False:
                        newtoken[u'analysis'][0][u'lex'] = \
                        analysis[i-1][u'analysis'][0][u'lex'] + u'-' + newtoken[u'analysis'][0][u'lex']
                    else:
                        newtoken[u'analysis'][0][u'lex'] = \
                        analysis[i-1][u'text'] + u'-' + newtoken[u'analysis'][0][u'lex']
                newtoken[u'text'] = analysis[i-1][u'text']+u'-'+newtoken[u'text']
                del analysis[i-1:i+2]
                analysis.insert(i-1, newtoken)
                i -= 1
            #except (IndexError, KeyError):
            #    print analysis
            i += 1
            
    for token in analysis:
        try:
            if u'analysis' not in token:
                if len(token[u'text']) > 2 or token[u'text'].isnumeric():
                    return True
            elif not token[u'analysis']:
                if len(token[u'text']) > 2:
                    return True
            elif "".join(takewhile(lambda c: c.isalpha(), token[u'analysis'][0][u'gr'])) in poslist\
                    and len(token[u'text']) > 2:
                    return True
        except (IndexError, KeyError):
            print(token)

    return False

