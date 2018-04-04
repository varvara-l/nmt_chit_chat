import argparse
import codecs
import json
import os
import sys
import time

import numpy as np
import requests
import tensorflow as tf

from . import attention_model
from . import gnmt_model
from . import model as nmt_model
from . import model_helper
from .nmt import add_arguments, create_hparams, create_or_load_hparams
from .utils import misc_utils as utils
from .utils import nmt_utils


class ConvAISampleBot:
    def __init__(self):
        self.chat_id = None
        self.observation = None
        self.engine = NMTModel()

    def observe(self, m):
        print("Observe:")
        if self.chat_id is None:
            if m['message']['text'].startswith('/start '):
                self.chat_id = m['message']['chat']['id']
                self.observation = m['message']['text']
                print("\tStart new chat #%s" % self.chat_id)
            else:
                self.observation = None
                print("\tChat not started yet. Ignore message")
        else:
            if self.chat_id == m['message']['chat']['id']:
                if m['message']['text'] == '/end':
                    self.observation = None
                    print("\tEnd chat #%s" % self.chat_id)
                    self.chat_id = None
                else:
                    self.observation = m['message']['text']
                    print("\tAccept message as part of chat #%s: %s" % (self.chat_id, self.observation))
            else:
                self.observation = None
                print("\tOnly one chat is allowed at a time. Ignore message from different chat #%s" %
                      m['message']['chat']['id'])
        return self.observation

    def act(self):
        print("Act:")
        if self.chat_id is None:
            print("\tChat not started yet. Do not act.")
            return

        if self.observation is None:
            print("\tNo new messages for chat #%s. Do not act." % self.chat_id)
            return

        message = {
            'chat_id': self.chat_id
        }

        text = self.engine(self.observation)

        data = {}
        if text == '':
            print("\tDecide to do not respond and wait for new message")
            return
        elif text == '/end':
            print("\tDecide to finish chat %s" % self.chat_id)
            self.chat_id = None
            data['text'] = '/end'
            data['evaluation'] = {
                'quality': 0,
                'breadth': 0,
                'engagement': 0
            }
        else:
            print("\tDecide to respond with text: %s" % text)
            data = {
                'text': text,
                'evaluation': 0
            }

        message['text'] = json.dumps(data)
        return message


def load_hparams(hparams_file):
    """Load hparams from an existing model directory."""
    # hparams_file = os.path.join(model_dir, "hparams")
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


class NMTModel:
    def __init__(self):
        self.sess = tf.Session(graph=infer_model.graph, config=utils.get_config_proto())
        self.loaded_infer_model = model_helper.load_model(infer_model.model, ckpt, self.sess, "infer")
        self.new_dialogue = True
        self.cur_persona = []
        self.history = []
        self.init_answer = "Please start the conversation\n"

    def __call__(self, *args, **kwargs):
        user_input = kwargs.get("user_input")
        # start a new dialogue (i.e. discard the history)
        if user_input.startswith('/start '):
            self.new_dialogue = True
            self.history = []

        # pick a new persona
        if self.new_dialogue and FLAGS.persona:
            self.cur_persona = np.random.choice(personas, 1)[0]

        if self.new_dialogue and FLAGS.persona and FLAGS.persona_display:
            for p in self.cur_persona:
                print(p)
            print('\n')
        self.new_dialogue = False

        # input for encoder
        history_len = max(0, len(self.history) - context_size)
        infer_data = [' __EOU__ '.join(self.cur_persona + self.history[history_len:] + [user_input])]
        # print(infer_data)

        # initialize iterator
        self.sess.run(infer_model.iterator.initializer,
                      feed_dict={infer_model.src_placeholder: infer_data,
                                 infer_model.batch_size_placeholder: 1})

        # decode
        nmt_outputs, infer_summary = self.loaded_infer_model.decode(self.sess)
        if hparams.beam_width > 0:
            # get the top translation.
            nmt_outputs = nmt_outputs[0]

        # get text translation
        translation = nmt_utils.get_translation(
            nmt_outputs,
            sent_id=0,
            tgt_eos=hparams.eos,
            subword_option=hparams.subword_option)
        cur_answer = str(translation, 'utf-8')
        self.history.extend([user_input, cur_answer])
        cur_answer += '\n'
        return cur_answer


if __name__ == "__main__":
    # get arguments
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    nmt_parser.add_argument('--context_size', default=sys.maxsize, help='number of previous utterances '
                                                                        'to include in the source (default: all)')
    nmt_parser.add_argument('--persona', default=None, help='persona descriptions (no persona used if not specified)')
    nmt_parser.add_argument('--persona_display', action='store_true', help='show persona descriptions'
                                                                           'in the beginning of a dialogue')

    FLAGS, unparsed = nmt_parser.parse_known_args()
    default_hparams = create_hparams(FLAGS)
    hparams = create_or_load_hparams(
        FLAGS.out_dir, default_hparams, FLAGS.hparams_path, save_hparams=True)
    context_size = int(FLAGS.context_size)
    # hparams = load_hparams(FLAGS.hparams_path)
    # with codecs.getreader("utf-8")(tf.gfile.GFile(FLAGS.hparams_path, "rb")) as f:
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

    # load personas from a file
    personas = []
    if FLAGS.persona:
        one_pers = []
        for line in open(FLAGS.persona):
            line = line.strip('\n')
            if line == '':
                personas.append(one_pers)
                one_pers = []
            else:
                one_pers.append(line)

    # starting bot
    BOT_ID = os.environ.get('BOT_ID')

    if BOT_ID is None:
        raise Exception('You should enter your bot token/id!')

    BOT_URL = os.path.join('https://ipavlov.mipt.ru/nipsrouter/', BOT_ID)

    bot = ConvAISampleBot()

    while True:
        try:
            time.sleep(1)
            print("Get updates from server")
            res = requests.get(os.path.join(BOT_URL, 'getUpdates'))

            if res.status_code != 200:
                print(res.text)
                res.raise_for_status()

            print("Got %s new messages" % len(res.json()))
            for m in res.json():
                print("Process message %s" % m)
                bot.observe(m)
                new_message = bot.act()
                if new_message is not None:
                    print("Send response to server.")
                    res = requests.post(os.path.join(BOT_URL, 'sendMessage'),
                                        json=new_message,
                                        headers={'Content-Type': 'application/json'})
                    if res.status_code != 200:
                        print(res.text)
                        res.raise_for_status()
            print("Sleep for 1 sec. before new try")
        except Exception as e:
            print("Exception: {}".format(e))