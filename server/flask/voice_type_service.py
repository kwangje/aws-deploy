import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from speaker_embedding import preprocess_wav, VoiceEncoder

# from demo_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse, sys, os, json, codecs, pathlib, pickle
import torch


class _Voice_Type_Detect_Service:

    # parser = argparse.ArgumentParser(description="Test for voice-type detection.")
    # parser.add_argument("--dst", type=pathlib.Path)
    # parser.add_argument("--file", type=pathlib.Path)
    # parser.add_argument("--output", type=pathlib.Path)
    # args = parser.parse_args()

    model = None
    _instance = None

    encoder = VoiceEncoder()

    """
        embeds_child_123 
        embeds_adult_female 
        embeds_adult_male 
    """

    with open("./embeds/voice_type_embeds_child_123.pkl", "rb") as f:
        embeds_child_123 = pickle.load(f)
    with open("./embeds/voice_type_embeds_adult_female.pkl", "rb") as f:
        embeds_adult_female = pickle.load(f)
    with open("./embeds/voice_type_embeds_adult_male.pkl", "rb") as f:
        embeds_adult_male = pickle.load(f)

    def cal_similarity(
        self, embeds_child, embeds_adult_female, embeds_adult_male, test
    ):
        labels = ["child", "adult_female", "adult_male", "unknown"]

        sim_child = np.inner(embeds_child, test)
        sim_adult_fm = np.inner(embeds_adult_female, test)
        sim_adult_m = np.inner(embeds_adult_male, test)

        if (sim_child and sim_adult_fm and sim_adult_m) < 0.5:
            sim_unk = int(1.0)
        else:
            sim_unk = int(0)

        similarities = [sim_child, sim_adult_fm, sim_adult_m, sim_unk]
        # filtered_similarities = [val for val in similarities if val < 0.5]

        stats = dict(zip(labels, similarities))
        result = max(stats, key=stats.get)

        return stats, result, similarities

    def age_clf_single(self, path_audio_file):
        """
        :param audio_file (str): Path to audio file to predict
        :return answer (str): voice-type predicted by the model
        """
        # audio_folder = "./audio_data/sample/test"

        test_path = Path(path_audio_file)
        test_wav = preprocess_wav(test_path)
        test_embed = self.encoder.embed_utterance(test_wav)
        score, answer, sim = self.cal_similarity(
            self.embeds_child_123,
            self.embeds_adult_female,
            self.embeds_adult_male,
            test_embed,
        )
        return answer


def Voice_Type_Detect_Service():
    """Factory function for Voice_Type_Detect_Service class.
    :return _Voice_Type_Detect_Service._instance (_Voice_Type_Detect_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Voice_Type_Detect_Service._instance is None:
        _Voice_Type_Detect_Service._instance = _Voice_Type_Detect_Service()
    return _Voice_Type_Detect_Service._instance


if __name__ == "__main__":
    vtd = Voice_Type_Detect_Service()
    predicted_voice_type = vtd.age_clf_single(
        "/Users/kwang/Desktop/sr-iptv-proto/audio_data/age-clf-val/sample_pcm_2/0a3df1ca-338b-4a81-a6bc-efbc41617ad8.wav"
    )

    print(predicted_voice_type)
    # audio_file = vtd.args.file
    # predicted_voice_type = vtd.age_clf_single(audio_file)
    # print(predicted_voice_type)

