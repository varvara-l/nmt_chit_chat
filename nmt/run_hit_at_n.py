import tensorflow as tf
import argparse
import os
import sys
from nltk import wordpunct_tokenize
import re

from .nmt import add_arguments, create_hparams, create_or_load_hparams
from . import attention_model
from . import gnmt_model
from . import model as nmt_model
from . import model_helper
from .utils import misc_utils as utils


# in_file: test file in ParlAI format
# returns: src, tg and ref file
# ref file format:
# <dialogue number><tab><label (0/1)> --- correct/incorrect target lines
def convert_test(in_file, out_file_prefix, context_size=sys.maxsize, persona=True):

    EOU = ' __EOU__ '
    src_out_file = out_file_prefix + '.src'
    tg_out_file = out_file_prefix + '.tg'
    ref_out_file = out_file_prefix + '.ref'
    src_out = open(src_out_file, 'w')
    tg_out = open(tg_out_file, 'w')
    ref_out = open(ref_out_file, 'w')

    re_str = re.compile(r'\d (.*)')
    re_num = re.compile(r'^(\d+)')

    persona_list = []
    history = []
    turn_cnt = 0
    #again = False
    for i, line in enumerate(open(in_file)):
        if i % 1000 == 0:
            sys.stderr.write('.')
        try:
            cur_num = int(re_num.findall(line)[0])
            line = re_str.findall(line.strip('\n'))[0]
        except:
            print('Wrong string format:\n{}'.format(line))
            sys.exit()

        if cur_num == 1:
            persona_list = []
            history = []
        # persona
        if line.startswith('your persona'):
            # tokenize persona line
            line = ' '.join(wordpunct_tokenize(line))
            persona_list.append(line)
        # turn
        else:
            try:
                turn, variants = line.split('\t\t')
                turn_utt1, turn_utt2 = turn.split('\t')
            except ValueError:
                print('Wrong string format:\n{}'.format(line))
                sys.exit()
            variants = variants.split('|')
            src_str = []
            persona_str = EOU.join(persona_list)
            history_txt = EOU.join(history[max(0, len(history) - context_size):])

            if persona:
                src_str.append(persona_str)
            if history_txt != '':
                src_str.append(history_txt)
            src_str.append(turn_utt1)
            src_str = EOU.join(src_str)
            for var in variants:
                src_out.write('%s\n' % src_str)
                tg_out.write('%s\n' % var)
                if var == turn_utt2:
                    label = 1
                else:
                    label = 0
                ref_out.write('%i\t%i\n' % (turn_cnt, label))
            history.extend([turn_utt1, turn_utt2])
            turn_cnt += 1

    src_out.close()
    tg_out.close()
    ref_out.close()
    sys.stderr.write('\n')
    return src_out_file, tg_out_file, ref_out_file


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    nmt_parser.add_argument('--test', help="test set to score on")
    nmt_parser.add_argument('--context_size', default=sys.maxsize, help='number of previous utterances '
                                                                   'to include in the source (default: all)')
    nmt_parser.add_argument('--persona', action='store_true', help='include persona description in the source')
    FLAGS, unparsed = nmt_parser.parse_known_args()
    #print(FLAGS)
    #print(unparsed)

    default_hparams = create_hparams(FLAGS)
    hparams = create_or_load_hparams(
        FLAGS.out_dir, default_hparams, FLAGS.hparams_path, save_hparams=True)

    # define the model type
    if not hparams.attention:
        model_creator = nmt_model.Model
    elif hparams.attention_architecture == "standard":
        model_creator = attention_model.AttentionModel
    elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
        model_creator = gnmt_model.GNMTModel
    else:
        raise ValueError("Unknown model architecture")

    eval_model = model_helper.create_eval_model(model_creator, hparams)

    # retrieve the saved model
    ckpt = FLAGS.ckpt
    out_dir = FLAGS.out_dir
    if not ckpt:
        ckpt = tf.train.latest_checkpoint(out_dir)

    test_src_file, test_tg_file, test_ref_file = convert_test(FLAGS.test,
                                                              FLAGS.test + '.conv',
                                                              context_size=FLAGS.context_size,
                                                              persona=FLAGS.persona)

    with tf.Session(graph=eval_model.graph, config=utils.get_config_proto()) as sess:
        loaded_eval_model = model_helper.load_model(eval_model.model, ckpt, sess, "eval")

        # initialise iterator
        # TODO: write test files
        eval_iterator_feed_dict = {
            eval_model.src_file_placeholder: test_src_file,
            eval_model.tgt_file_placeholder: test_tg_file
        }
        sess.run(eval_model.iterator.initializer, feed_dict=eval_iterator_feed_dict)
        all_sent_loss = loaded_eval_model.eval_no_ref(sess)

        turn_pred, turn_ref = [], []
        prev_dial_num = None
        hit1, hit5, hit8, hit10 = 0, 0, 0, 0
        for ref_line, sent_loss in zip(open(test_ref_file), all_sent_loss):
            cur_dial_num, label = ref_line.strip('\n').split('\t')
            # new example
            if prev_dial_num is not None and cur_dial_num != prev_dial_num:
                # compute values for the previous example
                sorted_res = [ref for (pred, ref) in sorted(zip(turn_pred, turn_ref), key=lambda p, r: p)]
                assert sum(sorted_res) == 1, "Wrong sum: {}, full res: {}, dialogue number {}".format(
                    sum(sorted_res),
                    ' '.join([str(r) for r in sorted_res]),
                    prev_dial_num
                )
                if sorted_res[0] == 1:
                    hit1 += 1
                if sum(sorted_res[:5]) == 1:
                    hit5 += 1
                if sum(sorted_res[:8]) == 1:
                    hit8 += 1
                if sum(sorted_res[:10]) == 1:
                    hit10 += 1

                # discard previous example
                turn_pred, turn_ref = [], []

            turn_pred.append(sent_loss)
            turn_ref.append(label)
            prev_dial_num = cur_dial_num

        print('Hit@1: %f' % (hit1/len(all_sent_loss)))
        print('Hit@5: %f' % (hit5/len(all_sent_loss)))
        print('Hit@8: %f' % (hit8 / len(all_sent_loss)))
        print('Hit@10: %f' % (hit10 / len(all_sent_loss)))
        all_sent_loss = all_sent_loss[0]
        flat_loss = [a for l in all_sent_loss for a in l]
        print(hit1, hit5, hit8, hit10, len(flat_loss), all_sent_loss.shape)
        print(test_ref_file)