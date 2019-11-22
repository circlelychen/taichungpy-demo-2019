import sys
import logging
import random

from collections import defaultdict


from rasa.nlu.training_data.formats import MarkdownReader, MarkdownWriter
from google.cloud import translate_v2 as translate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def random_select_samples(training_examples, percent):
    samples_dict = defaultdict(list)
    for example in training_examples:
        intent = example.get("intent")
        samples_dict[intent].append(example)
    logger.info("total intents: {0}".format(list(samples_dict.keys())))

    selected_nlu_train = []
    for key in samples_dict:
        origin_size = len(samples_dict[key])
        if origin_size * percent > 20:
            selected_examples = random.sample(samples_dict[key],
                                              k=int(origin_size * percent))
        else:
            selected_examples = samples_dict[key]
        selected_size = len(selected_examples)
        selected_nlu_train.extend(selected_examples)
        logger.info("intent: {0}, origin size: {1}, "
                    "selected size: {2}".format(
                        key, origin_size, selected_size))
    return selected_nlu_train


def task(source, dest, cred_file, percent):

    # load Rasa NLU training data
    r = MarkdownReader()
    with open(source, "r") as fin:
        nlu = fin.read()

    nlu_train = r.reads(nlu)

    translate_client = translate.Client.from_service_account_json(cred_file)

    def trans(text):
        trans_text = translate_client.translate(text, source_language="en",
                                                target_language="zh-TW")
        logger.info(u'origin: {}, translated: {}'.format(
            example.text, trans_text['translatedText']))
        return trans_text['translatedText']

    nlu_train.training_examples = random_select_samples(
        nlu_train.training_examples, percent)
    for example in nlu_train.training_examples:
        example.text = trans(example.text)
        if example.get("entities"):
            for entity in example.get("entities"):
                entity["value"] = trans(entity['value'])

    # Generate Rasa NLU translated training data
    w = MarkdownWriter()
    w.dump(dest, nlu_train)


def main(argv):
    source = argv[1]
    dest = argv[2]
    cred_file = argv[3]
    percent = float(argv[4])
    task(source, dest, cred_file, percent)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        logger.info(
            'Usage: ./translate.py [source] [dest] creds percent')
        sys.exit(1)
    main(sys.argv)
