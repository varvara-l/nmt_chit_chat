import argparse
import tensorflow as tf
import json
import sys
import codecs

from .nmt import add_arguments, create_hparams, create_or_load_hparams
from . import attention_model
from . import gnmt_model
from . import model as nmt_model
from . import model_helper
from .utils import misc_utils as utils
from .utils import nmt_utils


def load_hparams(hparams_file):
  """Load hparams from an existing model directory."""
  #hparams_file = os.path.join(model_dir, "hparams")
  if tf.gfile.Exists(hparams_file):
    print("# Loading hparams from %s" % hparams_file)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
      try:
        hparams_values = json.load(f)
        hparams = tf.contrib.training.HParams(**hparams_values)
      except ValueError:
        print("  can't load hparams file")
        return None
    return hparams
  else:
    return None


if __name__ == "__main__":

    init_answer = "Please start the conversation\n"
    cur_answer = init_answer
    user_input = None
    history = []

    # get arguments
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    default_hparams = create_hparams(FLAGS)
    hparams = create_or_load_hparams(
        FLAGS.out_dir, default_hparams, FLAGS.hparams_path, save_hparams=True)
    #hparams = load_hparams(FLAGS.hparams_path)
    #with codecs.getreader("utf-8")(tf.gfile.GFile(FLAGS.hparams_path, "rb")) as f:
    #  try:
    #    hparams_values = json.load(f)
    #    hparams = tf.contrib.training.HParams(**hparams_values)
    #  except ValueError:
    #    print("can't load hparams file")
    #    sys.exit()

    # define the model type
    if not hparams.attention:
        model_creator = nmt_model.Model
    elif hparams.attention_architecture == "standard":
        model_creator = attention_model.AttentionModel
    elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
        model_creator = gnmt_model.GNMTModel
    else:
        raise ValueError("Unknown model architecture")

    infer_model = model_helper.create_infer_model(model_creator, hparams)

    # retrieve the saved model
    ckpt = FLAGS.ckpt
    out_dir = FLAGS.out_dir
    if not ckpt:
        ckpt = tf.train.latest_checkpoint(out_dir)

    with tf.Session(graph=infer_model.graph, config=utils.get_config_proto()) as sess:
        loaded_infer_model = model_helper.load_model(infer_model.model, ckpt, sess, "infer")

        new_dialogue = True
        exit_dialogue = False

        while not exit_dialogue:
            print(cur_answer)
            sys.exit()
            user_input = input(cur_answer)

            # start a new dialogue (i.e. discard the history)
            if user_input == '@new':
                new_dialogue = True
                history = []
                continue
            # shut down
            if user_input == '@exit':
                exit_dialogue = True
                continue

            # decode
            infer_data = [user_input]
            # initialize iterator
            sess.run(infer_model.iterator.initializer,
                     feed_dict={infer_model.src_placeholder: infer_data,
                                infer_model.batch_size_placeholder: 1}
                    )

            nmt_outputs, infer_summary = loaded_infer_model.decode(sess)
            if hparams.beam_width > 0:
                # get the top translation.
                nmt_outputs = nmt_outputs[0]

            # get text translation
            translation = nmt_utils.get_translation(
                nmt_outputs,
                sent_id=0,
                tgt_eos=hparams.eos,
                subword_option=hparams.subword_option)
            cur_answer = str(translation, 'utf-8') + "\n"
