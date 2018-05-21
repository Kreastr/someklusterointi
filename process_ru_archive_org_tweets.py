import subprocess
import glob
import os
import simplejson
import re
import time
from datetime import datetime

from pymystem3 import Mystem

any_ru_cyrillic = re.compile('[\u0410-\u044f\u0401\u0451]')
nothing_but_ru_en = re.compile('[^0-9A-Za-z\u0410-\u044f\u0401\u0451 #\-\n]')

m = Mystem()

# filter out all tweets with lang=ru from bzipped archives
print('Rough filtering based on metadata...')
processes = []
#for i in range(1, 32):
    #processes.append(subprocess.Popen('for file in $(find {0:02}/ -name *.bz2); do echo $file; bzip2 -c -d $file|fgrep \'"lang":"ru"\'>>{0:02}.ru.json; done'.format(i), shell=True))

print('Waiting for all processes to finish...')
for p in processes:
    p.communicate()

print('Pre-processing text...')
# extract timestamp, id and raw text from json
with open("all_ru_tweets_unsorted.txt", 'w') as f_out:
    for d in range(1, 32):
        path = "{:02}.ru.json".format(d)

        if os.path.isfile(path):
            print("File: %s" % path)
            with open(path) as f:
                for line in f:
                    try:
                        json_obj = simplejson.loads(line)
                    except simplejson.errors.JSONDecodeError:
                        print('Invalid line: {}!'.format(line))
                        continue

                    if type(json_obj) is not dict:
                        print("Invalid line {}!".format(path, line))
                        continue
                    
                    if 'text' not in json_obj:
                        print("No tweet text found in file {}'s line {}!".format(path, line))
                        continue

                    if 'id' not in json_obj:
                        print("No tweet id found in file {}'s line {}!".format(path, line))
                        continue
                    
                    if 'timestamp_ms' not in json_obj:

                        if 'created_at' not in json_obj:
                            print("No time found in file {}'s line {}!".format(path, line))
                            continue

                        # convert to unix time with milliseconds
                        tweet_timestamp = int(time.mktime(datetime.strptime(json_obj['created_at'], '%a %b %d %X +0000 %Y').timetuple()) * 1000)
                    else:
                        tweet_timestamp = int(json_obj['timestamp_ms'])


                    # remove all newlines
                    tweet_text = json_obj['text'].lower().replace('\n', ' ').replace('\r', ' ')
                    # remove retweet marker
                    tweet_text = re.sub('^rt ', '', tweet_text)
                    # remove mentions
                    tweet_text = re.sub('@[^ ]+ ', '', tweet_text)
                    # remove urls
                    tweet_text = re.sub('https?[^ ]+', '', tweet_text)
                    # remove special characters
                    tweet_text = re.sub(nothing_but_ru_en, ' ', tweet_text)

                    # ignore the tweet if there are no russian letters left
                    if not any_ru_cyrillic.search(tweet_text):
                        continue

                    tweet_text = ''.join(m.lemmatize(tweet_text))
                    tweet_text = tweet_text.strip()

                    f_out.write(str(tweet_timestamp))
                    f_out.write(' ')
                    f_out.write(str(json_obj['id']))
                    f_out.write(' ')
                    f_out.write(tweet_text)
                    f_out.write('\n')


print('Sorting output based on timestamp')
subprocess.call('sort -n all_ru_tweets_unsorted.txt > all_ru_tweets.txt', shell=True)
