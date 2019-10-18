import logging
import os
import re
import typing
from typing import Any, Dict, List, Optional, Text, Tuple
import numpy as np

from rasa.nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.tokenizers import Token
from rasa.nlu.training_data import Message, TrainingData

from .base import BIOLUEntityExtractor

logger = logging.getLogger(__name__)

try:
    import keras as ks
    # avoid warning println on contrib import - remove for tf 2
except ImportError:
    ks = None


class KerasBaseEntityExtractor(BIOLUEntityExtractor):

    provides = ["entities"]

    requires = ["tokens"]

    language_list = ["zh"]

    defaults = {
        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        # "BILOU_flag": True,
        "token": "word",
        "strategy": "bi-lstm",
        "epochs": 100,
        "padding_size": 32
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        ent_tagger: Optional["Sequential"] = None,
        word_set: Optional["Set"] = None, label_set: Optional["Set"] = None,
        word2ind: Optional["Dict"] = None,
        ind2word: Optional["Dict"] = None,
        label2ind: Optional["Dict"] = None,
        ind2label: Optional["Dict"] = None,
    ) -> None:

        super(KerasBaseEntityExtractor, self).__init__(component_config)

        self._word_set = word_set
        self._label_set = label_set

        self._word2ind = word2ind
        self._ind2word = ind2word
        self._label2ind = label2ind
        self._ind2label = ind2label

        self.ent_tagger = ent_tagger

        self._check_keras()

    @staticmethod
    def _check_keras():
        if ks is None:
            raise ImportError(
                "Failed to import `Keras`. "
                "Please install `keras`. "
                "For example with `pip install keras`."
            )

    @classmethod
    def required_packages(cls):
        return ["keras"]

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["KerasBaseEntityExtractor"] = None,
        **kwargs: Any
    ) -> "KerasBaseEntityExtractor":
        import pickle
        from keras.models import load_model
        file_name = meta.get("filename", None)
        strategy = meta.get("strategy", None)
        if not file_name or not strategy:
            raise ValueError("Invalid model on loading.")

        # load assets
        with open(os.path.join(model_dir, file_name + "_word_set.pkl"), 'rb') as fin:
            word_set = pickle.load(fin)
        with open(os.path.join(model_dir, file_name + "_label_set.pkl"), 'rb') as fin:
            label_set = pickle.load(fin)
        with open(os.path.join(model_dir, file_name + "_word2ind.pkl"), 'rb') as fin:
            word2ind = pickle.load(fin)
        with open(os.path.join(model_dir, file_name + "_ind2word.pkl"), 'rb') as fin:
            ind2word = pickle.load(fin)
        with open(os.path.join(model_dir, file_name + "_label2ind.pkl"), 'rb') as fin:
            label2ind = pickle.load(fin)
        with open(os.path.join(model_dir, file_name + "_ind2label.pkl"), 'rb') as fin:
            ind2label = pickle.load(fin)

        # load keras model
        model_file_name = os.path.join(
            model_dir, file_name + "_ner_keras_based_weights.pkl")
        if os.path.exists(model_file_name):
            ent_tagger = KerasBaseEntityExtractor.config_model(
                strategy, len(word_set), len(label_set))
            ent_tagger.load_weights(model_file_name)
            return cls(meta, ent_tagger,
                       word_set=word_set,
                       label_set=label_set,
                       word2ind=word2ind,
                       ind2word=ind2word,
                       label2ind=label2ind,
                       ind2label=ind2label
                       )
        else:
            return cls(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        import pickle
        if self.ent_tagger:
            # save assets
            targets = [self._word_set, self._label_set, self._word2ind,
                       self._ind2word, self._label2ind, self._ind2label]
            archives = ["word_set", "label_set", "word2ind",
                        "ind2word", "label2ind", "ind2label"]
            for target, archive in zip(targets, archives):
                archive_filename = file_name + "_" + archive + ".pkl"
                asset_file_name = os.path.join(model_dir, archive_filename)
                with open(asset_file_name, 'wb') as fout:
                    pickle.dump(target, fout)

            # save keras model
            model_file_name = os.path.join(
                model_dir, file_name + "_ner_keras_based_weights.pkl")
            self.ent_tagger.save_weights(model_file_name)
            return {"filename": file_name,
                    "strategy": self.component_config["strategy"]}

    def _create_dataset(
        self, examples: List[Message]
    ) -> List[List[Tuple[Text, Text]]]:
        dataset = []
        for example in examples:
            entity_offsets = self._convert_example(example)
            data = self._from_json_to_dl(example, entity_offsets)
            dataset.append(data)
        return dataset

    @classmethod
    def config_model(cls, strategy, word_count, label_count) -> Optional["Sequential"]:

        if strategy not in ["bi-lstm-crf", "bi-lstm"]:
            raise NotImplementedError(
                "Only support strategy: [bi-lstm], [bi-lstm-crf]")

        ent_tagger = ks.Sequential()

        # add character embedding layer
        ent_tagger.add(ks.layers.Embedding(
            word_count, 300,
            mask_zero=True))
        ent_tagger.add(ks.layers.Bidirectional(
            ks.layers.LSTM(300, return_sequences=True)))
        ent_tagger.add(ks.layers.Activation('relu'))
        ent_tagger.add(ks.layers.Dropout(0.1))

        if strategy == "bi-lstm-crf":
            # add crf layer
            from keras_contrib.layers.crf import CRF
            ent_tagger.add(
                CRF(label_count,
                    learn_mode='join',
                    test_mode='viterbi',
                    sparse_target=False))

            from keras_contrib.losses import crf_loss
            from keras_contrib.metrics import crf_viterbi_accuracy
            loss_func = crf_loss
            metrics = [crf_viterbi_accuracy]
        elif strategy == "bi-lstm":
            ent_tagger.add(ks.layers.TimeDistributed(
                ks.layers.Dense(label_count)))
            ent_tagger.add(ks.layers.Activation('sigmoid'))

            loss_func = 'categorical_crossentropy'
            metrics = ["accuracy"]

        ent_tagger.compile(
            optimizer='adam',
            loss=loss_func,
            metrics=metrics)

        return ent_tagger

    @staticmethod
    def _convert_example(example: Message) -> List[Tuple[int, int, Text]]:
        def convert_entity(entity):
            return entity["start"], entity["end"], entity["entity"]

        return [convert_entity(ent) for ent in example.get("entities", [])]

    def _from_text_to_dl(
        self, tokens: List[Token], entities: List[Text] = None
    ) -> List[Tuple[Text, Text]]:
        """Takes a sentence and switches it to list-of-tuples format."""

        c2t = []
        for i, token in enumerate(tokens):
            # TBD: entity as label indices
            tag = entities[i] if entities else "N/A"
            c2t.append((token.text, tag))
        return c2t

    def _from_text_to_tokens(self, message: Message) -> List[Token]:
        if self.component_config["token"] == "word":
            return message.get("tokens")
        else:
            return self._from_text_to_chars(message)

    def _from_text_to_chars(self, message: Message) -> List[Token]:
        tokens = []
        for i, character in enumerate(list(re.sub(r"\s+", "", message.text))):
            tokens.append(Token(character, i))
        return tokens

    def _from_json_to_dl(
        self, message: Message, entity_offsets: List[Tuple[int, int, Text]]
    ) -> List[Tuple[Text, Text]]:
        """Convert json examples to format of underlying keras."""

        tokens = self._from_text_to_tokens(message)
        ents = self._bilou_tags_from_offsets(tokens, entity_offsets)
        return self._from_text_to_dl(tokens, ents)

    def _from_dl_to_json(
        self, tokens: List[Token], entities: List[Any]
    ) -> List[Dict[Text, Any]]:

        if len(tokens) != len(entities):
            raise Exception(
                "Inconsistency in amount of tokens between crfsuite and message"
            )

        return self._convert_bilou_tagging_to_entity_result(tokens, entities)

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        # checks whether there is at least one example with an entity annotation
        if training_data.entity_examples:
            # filter out pre-trained entity examples
            filtered_entity_examples = self.filter_trainable_entities(
                training_data.training_examples
            )

            # convert the dataset into features
            # this will train on ALL examples, even the ones
            # without annotations
            dataset = self._create_dataset(filtered_entity_examples)

            self._init_word_label_set(dataset)

            self._train_model(dataset)

    def _init_word_label_set(
        self,
        dataset: List[List[Tuple[Text, Text]]]
    ):
        self._word_set = set()
        self._word_set.add('PAD')
        self._word_set.add('UNK')
        self._word_set.add('CLS')
        self._word_set.add('SEP')

        self._label_set = set()
        self._label_set.add('PAD')

        for data in dataset:
            for token in data:
                self._word_set.add(token[0])
                self._label_set.add(token[1])
        self._word2ind = {word: index for index,
                          word in enumerate(self._word_set)}
        self._ind2word = {index: word for index,
                          word in enumerate(self._word_set)}
        self._label2ind = {label: index for index,
                           label in enumerate(self._label_set)}
        self._ind2label = {index: label for index,
                           label in enumerate(self._label_set)}

    def process(self, message: Message, **kwargs: Any) -> None:

        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set(
            "entities", message.get("entities", []) + extracted, add_to_output=True
        )

    def extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Take a sentence and return entities in json format"""

        if self.ent_tagger is not None:

            tokens = self._from_text_to_tokens(message)
            text_data = self._from_text_to_dl(tokens)
            features = self._sentence_to_features(text_data)
            labels = self.ent_tagger.predict([features])
            ents = []
            for index, label in enumerate(labels[:len(text_data)]):
                ent = {self._ind2label[idx]: prob for idx,
                       prob in enumerate(label[0])}
                ents.append(ent)
            return self._from_dl_to_json(tokens, ents)
        else:
            return []

    def _sentence_to_features(
        self,
        sentence: List[Tuple[Text, Text]],
    ) -> List[Dict[Text, Any]]:
        features = []
        for token in sentence:
            feature = self._word2ind['UNK']
            if token[0] in self._word_set:
                feature = self._word2ind[token[0]]
            features.append(feature)

        # padding
        padded_features = []
        for f in features:
            padded_features.append(f)
        for i in range(len(features), self.component_config["padding_size"]):
            padded_features.append(self._word2ind['PAD'])
        return padded_features

    def _sentence_to_labels(
        self,
        sentence: List[Tuple[Text, Text]],
    ) -> List[Text]:
        features = []
        for token in sentence:
            feature = self._label2ind[token[1]]
            features.append(feature)

        # padding
        padded_features = []
        for f in features:
            padded_features.append(f)
        for i in range(len(features), self.component_config["padding_size"]):
            padded_features.append(self._label2ind['PAD'])
        return padded_features

    @staticmethod
    def _turn_one_hot(total_list, size):
        one_hot_list = []
        for labels in total_list:
            tmp_list = []
            for label in labels:
                single_vec = np.zeros(size)
                single_vec[label] = 1
                tmp_list.append(single_vec)
            one_hot_list.append(tmp_list)
        return np.asarray(one_hot_list)

    def _train_model(
        self,
        df_train: List[
            List[Tuple[Optional[Text], Optional[Text], Text, Dict[Text, Any]]]
        ],
    ) -> None:
        """Train the crf tagger based on the training data."""
        x_train = np.asarray(
            [self._sentence_to_features(sent) for sent in df_train])
        y_train = np.asarray([self._sentence_to_labels(sent) for sent in
                              df_train])

        self.ent_tagger = KerasBaseEntityExtractor.config_model(
            self.component_config["strategy"], len(self._word_set), len(self._label_set))

        y_train = self._turn_one_hot(y_train, len(self._label_set))

        self.ent_tagger.fit(
            x_train, y_train,
            epochs=self.component_config["epochs"])
