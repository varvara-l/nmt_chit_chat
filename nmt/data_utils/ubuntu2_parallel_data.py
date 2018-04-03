from argparse import ArgumentParser
import csv


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input', help='input file')
    parser.add_argument('output_prefix', help='prefix for output files. The script will generate two files: '
                                              '<output_prefix>.src and <output_prefix>.tg')
    args = parser.parse_args()

    # extract context and ground truth from the initial answer
    all_contexts = []
    all_gr_truth = []
    reader = csv.reader(open(args.input))
    for line in reader:
        # adding strip() because some contexts and answers have it
        all_contexts.append(line[0].strip('"'))
        all_gr_truth.append(line[1].strip('"'))

    # context format: utterance1, utterance2, utterance3, utterance4
    # ground truth answer format: utterance5
    # we need to convert that to a parallel file of format:
    #       utterance1 --- utterance2
    #       utterance1 utterance2 --- utterance3
    #       utterance1 utterance2 utterance3 --- utterance4
    #       utterance1 utterance2 utterance3 utterance4 --- utterance5
    src_out = open(args.output_prefix + '.src', 'w')
    tg_out = open(args.output_prefix + '.tg', 'w')
    for context, answer in zip(all_contexts, all_gr_truth):
        utterances = [l for l in [st.strip() for st in context.split('__eot__')] if len(l) > 0]
        utterances += [answer]
        for i in range(1, len(utterances)):
            src_out.write('%s __eot__\n' % ' __eot__ '.join(utterances[0:i]))
            tg_out.write('%s __eot__\n' % utterances[i])
    src_out.close()
    tg_out.close()