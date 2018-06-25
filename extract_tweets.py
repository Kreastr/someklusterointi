import glob
import re
import time
from datetime import datetime

from pymystem3 import Mystem
any_ru_cyrillic = re.compile('[\u0410-\u044f\u0401\u0451]'.decode('unicode-escape'))
nothing_but_ru_en = re.compile('[^0-9A-Za-z\u0410-\u044f\u0401\u0451 #\-\n]'.decode('unicode-escape'))
nothing_but_ru_en_2 = re.compile('[0-9A-Za-z\u0410-\u044f]'.decode('unicode-escape'))

f_out = open("out.ru.txt", 'w')

m = Mystem()
#"screen_name":"

def grepContentTag(line,tag):
    start_index = line.find(tag) + len(tag)
    i = start_index

    while  i < len(line):
        if line[i] == '"' and line [i-1] != '\\':
            break
        i += 1
    return line[start_index:i].replace('\\n', ' ').replace('\\r', ' ')


files = glob.glob("*.ru.json")
files.sort()
print(files)

for filename in files:
    tweets = []
    print(filename)
    f = open(filename)

    for line in f:

            
        rawtext = grepContentTag(line,'"text":"').replace(',', ' ').decode('unicode-escape')
        
        # ignore the tweet if there are no russian letters left
        if not any_ru_cyrillic.search(rawtext):
            continue
        # remove all newlines
        tweet_text = rawtext.lower().replace('\n', ' ').replace('\r', ' ')
        # remove retweet marker
        tweet_text = re.sub('^rt ', '', tweet_text)
        # remove mentions
        tweet_text = re.sub('@[^ ]+ ', '', tweet_text)
        # remove urls
        tweet_text = re.sub('https?[^ ]+', '', tweet_text)
        # remove special characters
        tweet_text = re.sub(nothing_but_ru_en, ' ', tweet_text)
        
        lemmatized = ''
        for lemma in m.lemmatize(tweet_text.strip()):
            if nothing_but_ru_en_2.search(lemma):
                lemmatized += ''.join(filter(lambda c: nothing_but_ru_en_2.search(c) , lemma.lower()))+' '
        timestr = grepContentTag(line,'"created_at":"')
        #print (lemmatized)
        unix_tweet_time =int(time.mktime(datetime.strptime(timestr, '%a %b %d %X +0000 %Y').timetuple()) * 1000)
        tweets.append([unix_tweet_time,[timestr,grepContentTag(line,'"id_str":"'),grepContentTag(line,'"screen_name":"'),rawtext.encode('unicode-escape'),lemmatized.encode('unicode-escape')]])
        
    f.close()

    tweets.sort(key= lambda x: x[0])

    for outl in tweets:
        f_out.write(",".join(outl[1])+'\n')

f_out.close()
