import os.path
import re
import time
from datetime import datetime

# match tweet inside "text":"<TWEET>" which can contain escaped quotes
tweet_re = re.compile('"text":"((?:[^"\\\\]*(?:\\\\.)?)*)"')
tweet_id_re = re.compile('"id":(\d*)')
# timestamp was not added until 2014, fall back to using 'created at' if not available
timestamp_re = re.compile('"timestamp_ms":"(\d*)"')
created_at_re = re.compile('"created_at":"([^"]*)"')

parent_id_re = re.compile('"in_reply_to_status_id":(\d*),')

# id, id of highest level parent
tweet_parent = {}

with open("tweet_replies.txt", 'w') as f_out:
    for m in range(1, 13):
        for d in range(1, 32):

            path = "%02d/%02d.ru.json" % (m, d)

            if os.path.isfile(path):
                print("File: %s" % path)
                with open(path) as f:
                    for line in f:
                        text_match = tweet_re.search(line)
                        if not text_match:
                            print("No tweet text found in file %s's line %s!" % (path, line))
                            continue

                        id_match = tweet_id_re.search(line)
                        if not id_match:
                            print("No tweet id found in file %s's line %s!" % (path, line))
                            continue
                        
                        timestamp_match = timestamp_re.search(line)
                        if not timestamp_match:

                            created_at_match = created_at_re.search(line)
                            if not created_at_match:
                                print("No time found in file %s's line %s!" % (path, line))
                                continue

                            # convert to unix time with milliseconds
                            tweet_timestamp = int(time.mktime(datetime.strptime(created_at_match.group(1), '%a %b %d %X +0000 %Y').timetuple()) * 1000)
                        else:
                            tweet_timestamp = int(timestamp_match.group(1))


                        tweet_id = int(id_match.group(1))
                        tweet_text = text_match.group(1).replace('\\n', ' ').replace('\\r', ' ').decode('unicode-escape')


                        parent_match = parent_id_re.search(line)

                        if parent_match:
                            parent_id = int(parent_match.group(1))

                            while True:
                                # if the parent doesn't exists make a new chain
                                if parent_id not in tweet_parent:
                                    tweet_parent[tweet_id] = 0
                                    parent_id = tweet_id
                                    break

                                if tweet_parent[parent_id] == 0:
                                    break

                                parent_id = tweet_parent[parent_id]

                        else:
                            tweet_parent[tweet_id] = 0
                            parent_id = tweet_id

                        f_out.write(str(tweet_timestamp))
                        f_out.write(' ')
                        f_out.write(str(parent_id))
                        f_out.write(' ')
                        f_out.write(tweet_text.encode('utf-8'))
                        f_out.write('\n')
