from vecs import Vecs
import csv
import sys
import cPickle as pickle

sys.path.append('/home/aminoff/SVD')
import translation
import fasttext

class WordVectorTranslator():
    
    def __init__(self, force_update=False):
        if force_update:
            twitter_vecs = Vecs('vocab.txt', 'vecs.bin')
            s24_vecs = Vecs('s24_swivel_vocab.txt', 's24_swivel.bin')
             
            with open('/home/aminoff/wikitranslation/fi-ru_symmetric_pairs_wiktionary_lren.csv') as f:
                word_pairs = [a for a in csv.reader(f)][1:]
                word_pairs_reversed = [[a[1], a[0]] for a in word_pairs][1:]

            fi_mat, ru_mat = translation.make_training_matrices(s24_vecs, twitter_vecs, word_pairs)
            self.fi_translation_mat = translation.learn_transformation(fi_mat, ru_mat, normalize_vectors=True)

            with open('fi_ru_translation_matrix.bin', 'w') as f:
                pickle.dump(self.fi_translation_mat, f)

        else:
            with open('fi_ru_translation_matrix.bin') as f:
                self.fi_translation_mat = pickle.load(f)


    def translate_vec(self, vector):
        return vector.dot(self.fi_translation_mat)

