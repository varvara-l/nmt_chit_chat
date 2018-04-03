from . import gnmt_model


class TwoDecoderModel(gnmt_model.GNMTModel):

    def __init__(self,
                 hparams,
                 mode,
                 iterator,
                 source_vocab_table,
                 target_vocab_table,
                 reverse_target_vocab_table=None,
                 scope=None,
                 extra_args=None):
        super(TwoDecoderModel, self).__init__(
            hparams=hparams,
            mode=mode,
            iterator=iterator,
            source_vocab_table=source_vocab_table,
            target_vocab_table=target_vocab_table,
            reverse_target_vocab_table=reverse_target_vocab_table,
            scope=scope,
            extra_args=extra_args)

