import tensorflow as tf
import argparse
import os
import sys

from .nmt import add_arguments, create_hparams, create_or_load_hparams
from . import attention_model
from . import gnmt_model
from . import model as nmt_model
from . import model_helper
from .utils import misc_utils as utils


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    nmt_parser.add_argument('--context_size', default=sys.maxsize, help='number of previous utterances '
                                                                   'to include in the source (default: all)')
    nmt_parser.add_argument('--persona', action='store_true', help='include persona description in the source')
    FLAGS, unparsed = nmt_parser.parse_known_args()
    #print(FLAGS)
    #print(unparsed)
    '''
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

    infer_model = model_helper.create_eval_model(model_creator, hparams)

    # retrieve the saved model
    ckpt = FLAGS.ckpt
    out_dir = FLAGS.out_dir
    if not ckpt:
        ckpt = tf.train.latest_checkpoint(out_dir)

    with tf.Session(graph=infer_model.graph, config=utils.get_config_proto()) as sess:
        loaded_eval_model = model_helper.load_model(infer_model.model, ckpt, sess, "infer")
    '''