# coding=utf-8
from components.extractor import KerasBaseEntityExtractor
import pytest
import os
import logging

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData, Message
from rasa.nlu.tokenizers import Token

logger = logging.getLogger(__name__)

PADDING_SIZE = 16


@pytest.fixture(scope="session")
def examples():  # {{{
    m1 = Message("5000台幣等值多少日幣",
                 {
                     "intent": "匯率兌換",
                     "entities": [
                         {"start": 0, "end": 4, "value": "5000", "entity":
                          "money"},
                         {"start": 4, "end": 6, "value": "台幣", "entity":
                          "currency"},
                         {"start": 10, "end": 12, "value": "日幣", "entity":
                          "currency"}
                     ]
                 })
    m1.set("tokens", [
        Token("5000", 0, {"ckip_pos_tag": "Neu"}),
        Token("台幣", 4, {"ckip_pos_tag": "Na"}),
        Token("等值", 6, {"ckip_pos_tag": "VH"}),
        Token("多少", 8, {"ckip_pos_tag": "Neqa"}),
        Token("日幣", 10, {"ckip_pos_tag": "Na"})
    ])
    m2 = Message("兩千台幣可以換多少日圓",
                 {
                     "intent": "匯率兌換",
                     "entities": [
                         {
                             "start": 0,
                             "end": 2,
                             "value": "兩千",
                             "entity": "money"
                         },
                         {
                             "start": 2,
                             "end": 4,
                             "value": "台幣",
                             "entity": "currency"
                         },
                         {
                             "start": 9,
                             "end": 11,
                             "value": "日圓",
                             "entity": "currency"
                         }
                     ]
                 })
    m2.set("tokens", [
        Token("兩千", 0, {"ckip_pos_tag": "Neu"}),
        Token("台幣", 2, {"ckip_pos_tag": "Na"}),
        Token("可以", 4, {"ckip_pos_tag": "D"}),
        Token("換", 6, {"ckip_pos_tag": "VC"}),
        Token("多少", 7, {"ckip_pos_tag": "Neqa"}),
        Token("日圓", 9, {"ckip_pos_tag": "Nf"})
    ])
    examples = [m1, m2]
    return examples
    # }}}


def test_create_dataset(examples):
    ext = KerasBaseEntityExtractor(
        component_config={"epochs": 3, "padding_size": PADDING_SIZE})

    filtered_entity_examples = ext.filter_trainable_entities(examples)
    dataset = ext._create_dataset(filtered_entity_examples)
    for d in dataset:
        logger.info(d)


def test_gen_words_and_labels_set(examples):
    ext = KerasBaseEntityExtractor(
        component_config={"token": "char", "epochs": 3, "padding_size": PADDING_SIZE})

    filtered_entity_examples = ext.filter_trainable_entities(examples)
    dataset = ext._create_dataset(filtered_entity_examples)
    ext._init_word_label_set(dataset)
    assert ext._ind2word[ext._word2ind['多']] == '多'
    assert ext._word2ind[ext._ind2word[0]] == 0
    assert ext._ind2label[ext._label2ind['L-money']] == 'L-money'
    assert ext._label2ind[ext._ind2label[0]] == 0

    ext = KerasBaseEntityExtractor(
        component_config={"token": "word", "epochs": 3, "padding_size": PADDING_SIZE})

    filtered_entity_examples = ext.filter_trainable_entities(examples)
    dataset = ext._create_dataset(filtered_entity_examples)
    ext._init_word_label_set(dataset)
    assert ext._ind2word[ext._word2ind['台幣']] == '台幣'
    assert ext._word2ind[ext._ind2word[0]] == 0
    assert ext._ind2label[ext._label2ind['U-money']] == 'U-money'
    assert ext._label2ind[ext._ind2label[0]] == 0


def test_sentence_to_features(examples):
    ext = KerasBaseEntityExtractor(
        component_config={"epochs": 3, "padding_size": PADDING_SIZE})

    filtered_entity_examples = ext.filter_trainable_entities(examples)
    dataset = ext._create_dataset(filtered_entity_examples)
    ext._init_word_label_set(dataset)

    len(dataset[0])
    padded_features = ext._sentence_to_features(dataset[0])
    assert len(padded_features) == PADDING_SIZE
    assert len(set(padded_features[len(dataset[0]):])) == 1


def test_persist_then_load_lstm(examples):

    ext = KerasBaseEntityExtractor(
        component_config={"epochs": 3, "padding_size": PADDING_SIZE})
    ext.train(TrainingData(training_examples=examples), RasaNLUModelConfig())

    import tempfile
    file_name = "lstm_tmp"
    dir_path = tempfile.mkdtemp()
    meta = ext.persist(file_name, dir_path)
    assert meta.get("filename", None) == file_name

    loaded_ext = ext.load(meta, dir_path)
    import shutil
    shutil.rmtree(dir_path)

    assert ext._word_set == loaded_ext._word_set
    assert ext._label_set == loaded_ext._label_set
    assert ext._word2ind == loaded_ext._word2ind
    assert ext._ind2word == loaded_ext._ind2word
    assert ext._label2ind == loaded_ext._label2ind
    assert ext._ind2label == loaded_ext._ind2label

    extracted = ext.extract_entities(examples[1])
    logger.info(extracted)


def test_persist_then_load_lstm_crf(examples):

    ext = KerasBaseEntityExtractor(
        component_config={
            "epochs": 3,
            "strategy": "bi-lstm-crf",
            "padding_size": PADDING_SIZE})
    ext.train(TrainingData(training_examples=examples), RasaNLUModelConfig())

    import tempfile
    file_name = "lstm_tmp"
    dir_path = tempfile.mkdtemp()
    meta = ext.persist(file_name, dir_path)
    assert meta.get("filename", None) == file_name

    loaded_ext = ext.load(meta, dir_path)
    import shutil
    shutil.rmtree(dir_path)

    assert ext._word_set == loaded_ext._word_set
    assert ext._label_set == loaded_ext._label_set
    assert ext._word2ind == loaded_ext._word2ind
    assert ext._ind2word == loaded_ext._ind2word
    assert ext._label2ind == loaded_ext._label2ind
    assert ext._ind2label == loaded_ext._ind2label

    extracted = ext.extract_entities(examples[1])
    logger.info(extracted)


def test_lstm_extractor(examples):
    ext = KerasBaseEntityExtractor(
        component_config={
            "token": "char",
            "strategy": "bi-lstm",
            "epochs": 3,
            "padding_size": PADDING_SIZE})
    ext.train(TrainingData(training_examples=examples), RasaNLUModelConfig())

    extracted = ext.extract_entities(examples[1])
    logger.info(extracted)
    # assert extracted[0]['value'] == '台幣'
    # assert extracted[0]['start'] == 2
    # assert extracted[0]['end'] == 4


def test_lstm_crf_extractor(examples):
    ext = KerasBaseEntityExtractor(
        component_config={"epochs": 3, "strategy": "bi-lstm-crf",
                          "padding_size": PADDING_SIZE})
    ext.train(TrainingData(training_examples=examples), RasaNLUModelConfig())

    extracted = ext.extract_entities(examples[1])
    logger.info(extracted)
    # assert extracted[0]['value'] == '台幣'
    # assert extracted[0]['start'] == 2
    # assert extracted[0]['end'] == 4
