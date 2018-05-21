# requires a file sorted by tweet chains

with open('tweet_replies_sorted2.txt') as f, open('combined_tweet_replies.txt', 'w') as f_out:
    current_chain_id = 0
    line = 0
    for l in f:
        line += 1
        if line % 10000 == 0:
            print(line)
        l = l.strip()
        parts = l.split(' ')
        chain_id = int(parts[1])

        if chain_id == current_chain_id:
            f_out.write(' ')
            f_out.write(' '.join(parts[2:]))
        else:
            f_out.write('\n')
            f_out.write(l)
            current_chain_id = chain_id

    f_out.write('\n')
    
