import argparse
import os
import random
import sys
import tensorflow as tf

from .nmt import create_or_load_hparams, create_hparams, add_arguments
from .utils import misc_utils as utils
from .utils import evaluation_utils


def two_dec_train():
    pass


def two_dec_inference():
    pass


def run_main(flags, default_hparams, train_fn, inference_fn, target_session=""):
  """Run main."""
  # Job
  jobid = flags.jobid
  num_workers = flags.num_workers
  utils.print_out("# Job id %d" % jobid)

  # Random
  random_seed = flags.random_seed
  if random_seed is not None and random_seed > 0:
    utils.print_out("# Set random seed to %d" % random_seed)
    random.seed(random_seed + jobid)
    np.random.seed(random_seed + jobid)

  ## Train / Decode
  out_dir = flags.out_dir
  if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)

  # Load hparams.
  hparams = create_or_load_hparams(
      out_dir, default_hparams, flags.hparams_path, save_hparams=(jobid == 0))

  if flags.inference_input_file:
    # Inference indices
    hparams.inference_indices = None
    if flags.inference_list:
      (hparams.inference_indices) = (
          [int(token)  for token in flags.inference_list.split(",")])

    # Inference
    trans_file = flags.inference_output_file
    ckpt = flags.ckpt
    if not ckpt:
      ckpt = tf.train.latest_checkpoint(out_dir)
    inference_fn(ckpt, flags.inference_input_file,
                 trans_file, hparams, num_workers, jobid)

    # Evaluation
    ref_file = flags.inference_ref_file
    if ref_file and tf.gfile.Exists(trans_file):
      for metric in hparams.metrics:
        score = evaluation_utils.evaluate(
            ref_file,
            trans_file,
            metric,
            hparams.subword_option)
        utils.print_out("  %s: %.1f" % (metric, score))
  else:
    # Train
    train_fn(hparams, target_session=target_session)

def main(unused_argv):
  default_hparams = create_hparams(FLAGS)
  train_fn = two_dec_train
  inference_fn = two_dec_inference
  run_main(FLAGS, default_hparams, train_fn, inference_fn)


if __name__ == "__main__":
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)