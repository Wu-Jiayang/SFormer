# -*- coding: utf-8 -*-
# @Time    : 2022-01-16 17:31
# @Author  : 吴佳杨

import paddle
from paddle import nn
from paddlenlp.transformers import UnifiedTransformerPretrainedModel
from paddlenlp.transformers.unified_transformer.modeling import UnifiedTransformerLMHead, UnifiedTransformerEmbeddings
from paddlenlp.transformers.model_utils import register_base_model
from paddle.fluid.data_feeder import convert_dtype


@register_base_model
class UnifiedStateTransformerModel(UnifiedTransformerPretrainedModel):
    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            normalize_before=True,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            unk_token_id=0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            mask_token_id=30000, ):
        super(UnifiedStateTransformerModel, self).__init__()
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.initializer_range = initializer_range

        self.embeddings = UnifiedTransformerEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers,
                                             encoder_norm)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                state,  # state_shape = (batch_size, state_size, hidden_size)
                use_cache=False,
                cache=None):
        assert attention_mask.dtype.name == 'FP32'
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        state_size = state.shape[1]     # .numpy()
        embedding_output = paddle.concat([state, embedding_output], axis=1)
        attention_mask = nn.Pad2D([state_size, 0, state_size, 0], mode='replicate')(attention_mask)
        if use_cache:
            if cache is None:
                cache = self.encoder.gen_cache(embedding_output)
            sequence_output, cache = self.encoder(embedding_output, attention_mask, cache)
            state = sequence_output[:, :state_size, :]
            sequence_output = sequence_output[:, state_size:, :]
            return sequence_output, cache, state
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            state = sequence_output[:, :state_size, :]
            sequence_output = sequence_output[:, state_size:, :]
            return sequence_output, state


class UnifiedStateTransformerLMHeadModel(UnifiedTransformerPretrainedModel):
    def __init__(self, unified_transformer):
        super(UnifiedStateTransformerLMHeadModel, self).__init__()
        self.unified_transformer = unified_transformer
        self.lm_head = UnifiedTransformerLMHead(
            self.unified_transformer.config["hidden_size"],
            self.unified_transformer.config["vocab_size"],
            self.unified_transformer.config["hidden_act"],
            self.unified_transformer.embeddings.word_embeddings.weight)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                state,
                masked_positions=None,
                use_cache=False,
                cache=None):
        outputs = self.unified_transformer(input_ids, token_type_ids,
                                           position_ids, attention_mask, state,
                                           use_cache, cache)
        if use_cache:
            sequence_output, cache, state = outputs
            logits = self.lm_head(sequence_output, masked_positions)
            return logits, cache, state
        else:
            sequence_output, state = outputs
            logits = self.lm_head(sequence_output, masked_positions)
            return logits, state

    def prepare_faster_entry(self, kwargs):
        from paddlenlp.ops import FasterUnifiedTransformer
        use_fp16_decoding = kwargs.get('use_fp16_decoding', False)
        decode_strategy = kwargs.get('decode_strategy')
        if decode_strategy == 'sampling' and kwargs.get(
                'top_k') != 0 and kwargs.get('top_p') != 1:
            raise AttributeError(
                    "Only topk sampling or topp sampling are supported. " \
                    "Topk sampling and topp sampling cannot be both applied in the faster version.")
        if kwargs['repetition_penalty'] != 1.0:
            # not support for repetition_penalty yet in the faster version
            raise AttributeError(
                "'repetition_penalty != 1' is not supported yet in the faster version"
            )
        if kwargs['forced_bos_token_id'] is not None:
            # not support for min_length yet in the faster version
            raise AttributeError(
                "'forced_bos_token_id != None' is not supported yet in the faster version"
            )
        self._faster_entry = FasterUnifiedTransformer(
            self, use_fp16_decoding=use_fp16_decoding).forward
        return self._faster_entry

    def adjust_logits_during_generation(self, logits):
        # pre-process distribution
        logits[:, self.unified_transformer.unk_token_id] = -1e9
        logits[:, self.unified_transformer.bos_token_id] = -1e9
        logits[:, self.unified_transformer.mask_token_id] = -1e9
        return logits

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      token_type_ids,
                                      position_ids,
                                      attention_mask,
                                      state,
                                      use_cache=False,
                                      cache=None,
                                      **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        if cache is not None and use_cache:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            attention_mask = attention_mask[:, :, -1, :].unsqueeze(2)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "state": state,
            "use_cache": use_cache,
            "cache": cache
        }

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return getattr(getattr(self, self.base_model_prefix), name)
            except AttributeError:
                try:
                    return getattr(self, self.base_model_prefix).config[name]
                except KeyError:
                    raise e

    @staticmethod
    def update_model_kwargs_for_generation(outputs,
                                           model_kwargs,
                                           is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache
        if model_kwargs["use_cache"]:
            model_kwargs["cache"] = outputs[1]

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], axis=-1)

        # update position_ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat(
                [position_ids, position_ids[:, -1].reshape((-1, 1)) + 1],
                axis=-1)

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # nn.Pad2D don't support the data type `bool`
            if convert_dtype(attention_mask.dtype) == 'bool':
                attention_mask = paddle.cast(attention_mask, 'int64')
            attention_mask = nn.Pad2D(
                [0, 0, 0, 1], mode='replicate')(attention_mask)
            attention_mask = nn.Pad2D([0, 1, 0, 0], value=-1e9)(attention_mask)
            dtype = convert_dtype(attention_mask.dtype)
            if 'int' in dtype:
                attention_mask[:, :, -1, -1] = 1
            elif 'float' in dtype:
                attention_mask[:, :, -1, -1] = 0.0
            else:
                raise ValueError('The data type of input `attention_mask` must '
                                 'be bool, int or float')
            model_kwargs["attention_mask"] = attention_mask

        return model_kwargs
