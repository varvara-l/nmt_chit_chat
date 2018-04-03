import csv
import glob
from argparse import ArgumentParser
import os
import sys
from nltk import wordpunct_tokenize
import numpy as np


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input', help='input directory')
    parser.add_argument('output_prefix', help='prefix for output files. The script will generate two files: '
                                              '<output_prefix>.src and <output_prefix>.tg')
    #parser.add_argument('--persona', action='store_true')
    #parser.add_argument('--context', default=sys.maxsize, help='max number of utterances from previous context '
    #                                                           'included in the source part')

    args = parser.parse_args()
    EOU = '__eou__'
    EOT = '__eot__'

    all_files = glob.glob(os.path.join(args.input, '*/*.tsv'))
    all_dialogues = []
    sys.stderr.write('Parsing files')
    for ii, a_file in enumerate(all_files):
        if ii % 10000 == 0:
            sys.stderr.write('.')

        if not os.path.isfile(a_file):
            continue
        reader = csv.reader(open(a_file), delimiter='\t')
        cur_dialogue = []

        prev_author = ''
        utt = []
        for line in reader:
            txt = ' '.join(wordpunct_tokenize(line[3].strip()))
            if line[1] == prev_author or prev_author == '':
                utt.append(txt)
            else:
                cur_dialogue.append(' __eou__ '.join(utt) + ' __eou__')
                utt = [txt]
            prev_author = line[1]
        if len(utt) != 0:
            cur_dialogue.append(' __eou__ '.join(utt) + ' __eou__')

        all_dialogues.append(cur_dialogue)
    #print(all_dialogues[100:101])
    all_lens = [len(cur_d) for cur_d in all_dialogues]
    sys.stderr.write('Min turns: %i, max turns: %i\n' % (min(all_lens), max(all_lens)))
    sys.stderr.write('Average: %f, std: %f\n' % (np.average(all_lens), np.std(all_lens)))
    #sys.exit()

    sys.stderr.write('\nWriting dialogues')
    src_out = open(args.output_prefix + '.src', 'w')
    tg_out = open(args.output_prefix + '.tg', 'w')
    for ii, dial in enumerate(all_dialogues):
        if ii % 10000 == 0:
            sys.stderr.write('.')
        for i in range(1, len(dial)):
            src_str = ' __eot__ '.join(dial[i-10:i]) + ' __eot__'
            tg_str = dial[i] + ' __eot__'
            src_out.write('%s\n' % src_str)
            tg_out.write('%s\n' % tg_str)
    src_out.close()
    tg_out.close()
    sys.stderr.write('\n')