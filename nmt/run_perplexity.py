import tensorflow as tf
import argparse
import os

from .nmt import add_arguments, create_hparams, create_or_load_hparams
from . import attention_model
from . import gnmt_model
from . import model as nmt_model
from . import model_helper
from .utils import misc_utils as utils


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
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

    with tf.Session(graph=eval_model.graph, config=utils.get_config_proto()) as sess:
        loaded_eval_model = model_helper.load_model(eval_model.model, ckpt, sess, "eval")

        # initialise iterator
        test_src_file = hparams.test_prefix + '.' + hparams.src
        test_tgt_file = hparams.test_prefix + '.' + hparams.tgt
        assert os.path.isfile(test_src_file)
        assert os.path.isfile(test_tgt_file)
        eval_iterator_feed_dict = {
            eval_model.src_file_placeholder: test_src_file,
            eval_model.tgt_file_placeholder: test_tgt_file
        }
        sess.run(eval_model.iterator.initializer, feed_dict=eval_iterator_feed_dict)

        perplexity = model_helper.compute_perplexity(loaded_eval_model, sess, 'my_eval')
        print('Perplexity: %f' % perplexity)