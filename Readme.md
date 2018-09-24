# Twitter Clustering

*Contributors: Onni Kosomaa, Aleksei Romanenko, Christoffer Aminoff*
*Supervisor: Prof. Jouko Vankka*

The Python data processing scripts are in this folder.

**visualization** contains the browser visualization.

**clustering.py** contains cluster generation.

**calculate_idfs.py** loads cached idfs from the json file given, or if force_recalc is specified calculates idfs from the text file. 
(A parameter should be added so that the JSON file and text file doesn't use the same parameter.

**cluster_exporter.py** contains all code to export cluster data into a format that the web visualization can read.

## Cluster generation

### Data pre-processing

The data is stored in text files with one document per line, with each line starting with a unix timestamp in milliseconds and a document id with spaces as separators.

For example: 1527492874000 123456789 Tweet tweet

Remember to match lemmatized word vectors with lemmatized documents.

### Cluster generation

Run **clustering.py**

* `-e, --embeddings=` Word embeddings in binary swivel format. (See Swivel documentation text2bin.py)
* `-v, --vocab=` Word embedding vocabulary. (See Swivel documentation)
* `-t, --text=` Text corpus in format specified in [Data pre-processing](#data-pre-processing)
* `-i, --idfs=` JSON file to load idfs from, format should be a dictionary with (word, weight) pairs.
* `-l, --lang=` Specify corpus language, for example 'fi' or 'ru'.

More configuration can be done in the main function in **clustering.py**.



### Exporting clusters

**cluster_exporter.py** convert_to_dict function converts a list of documents to a format ready to be saved into a JSON file for the visualization.

**cluster_exporter.py** save_cluster_texts fetches the original, un-processed tweets from the Internet Archive Twitter dataset folder for the given clusters. These can be used for the detailed tweet lists in the visualization.

## Visualization data format

The program has evolved quite a bit since the initial version of the data format was made, and could use some improvements.

The JSON consists of a list of 'snapshots', all of which have a 't' property which is a unix timestamp in seconds.

The 'n' property is a dictionary of 'new cluster' object which contains properties for a new cluster. The cluster id works as the key.

The 'u' property is a dictionary of update to existing clusters, which contain for example the new size. The cluster id works as the key.

```json
[
  {
    t: 1234567890,
    n: {
      123: {
        s: 2,
        k: ['cluster', 'key', 'words'],
        tags: ['cluster', 'tags', 'example']
        lang: 'fi'
        sentiment: 0.42,
        sentiment_total: 1.0,
        t_sne: [12.3, 4.56]
      }
    },
    
    u: {
      13: {
        s: 4,
        k: ['updated', 'cluster', 'keywords']
        sentiment: -0.2,
        sentiment_accum: -0.4
      }
    }
  }
]
```


## Improvements

* Remove references to Swivel's code.

