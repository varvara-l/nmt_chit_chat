from nltk import wordpunct_tokenize, FreqDist
import sys
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input', help='input corpus')
    parser.add_argument('output', help='new file to store vocabulary')
    parser.add_argument('--vocab_size', default=None, help='max vocabulary size (default: actual size of vocabulary)')
    parser.add_argument('--cutoff', default=1, help='min frequency of words in the vocabulary (default: 1)')
    args = parser.parse_args()
    fdist_ubuntu = FreqDist()
    for i, line in enumerate(open(args.input)):
        if i % 100000 == 0:
            sys.stderr.write('.')
        for word in wordpunct_tokenize(line):
            fdist_ubuntu[word.lower()] += 1

    out_file = open(args.output, 'w')
    voc_size = args.vocab_size if args.vocab_size else len(fdist_ubuntu)
    print voc_size
    #sys.exit()
    for w, f in fdist_ubuntu.most_common():
        #sys.stderr.write('%s\t%i\n' % (w, f))
        if f >= args.cutoff:
            sys.stderr.write('.')
            out_file.write('%s\n' % w)
    out_file.close()