import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from config import Preprocess_text_config, config
from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import (
    bert_models,
    cleaned_text_to_sequence,
    extract_bert_feature,
    clean_text,
)
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
from preprocess_text import preprocess 
from pathlib import Path

# このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
pyopenjtalk_worker.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()

preprocess_text_config: Preprocess_text_config = config.preprocess_text_config


def process_line(x: tuple[str, bool]):
    line, add_blank = x
    device = config.bert_gen_config.device
    if config.bert_gen_config.use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(
        phone, tone, Languages[language_str]
    )

    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")

    # for mood カッコ内を削除したものを取り寄せ
    use_jp_extra = hps.version.endswith("JP-Extra")
    rm_par_text, rm_par_phones, rm_par_tones, rm_par_word2ph, non_rm_norm_text, non_rm_phones, non_rm_tones, non_rm_word2ph, rm_par_fags = clean_text(
        text,
        Languages[language_str],
        use_jp_extra=use_jp_extra,
        raise_yomi_error=False,
    )

    rm_par_phones = commons.intersperse(rm_par_phones, 0)
    for i in range(len(non_rm_word2ph)):
        non_rm_word2ph[i] = non_rm_word2ph[i] * 2
    non_rm_word2ph[0] += 1
    for i in range(len(rm_par_word2ph)):
        rm_par_word2ph[i] = rm_par_word2ph[i] * 2
    rm_par_word2ph[0] += 1

    # try:
    #     bert = torch.load(bert_path)
    #     assert bert.shape[-1] == len(rm_par_phones)
    # except Exception:
    #     bert = extract_bert_feature(non_rm_norm_text, non_rm_word2ph, Languages[language_str], device, None, 0.7, rm_par_text,rm_par_word2ph, rm_par_fags)
    #     assert bert.shape[-1] == len(rm_par_phones)
    #     torch.save(bert, bert_path)
    bert = extract_bert_feature(
        non_rm_norm_text, 
        non_rm_word2ph, 
        Languages[language_str], 
        device, 
        None, 
        0.7, 
        rm_par_text,
        rm_par_word2ph, 
        rm_par_fags
        )
    assert bert.shape[-1] == len(rm_par_phones)
    torch.save(bert, bert_path)


#preprocess_text_config = config.preprocess_text_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    hps = HyperParameters.load_from_json(config_path)
    lines: list[str] = []
    with open(hps.data.training_files, "r", encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, "r", encoding="utf-8") as f:
        lines.extend(f.readlines())
    add_blank = [hps.data.add_blank] * len(lines)

    if len(lines) != 0:
        # pyopenjtalkの別ワーカー化により、並列処理でエラーがでる模様なので、一旦シングルスレッド強制にする
        num_processes = 1
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            _ = list(
                tqdm(
                    executor.map(process_line, zip(lines, add_blank)),
                    total=len(lines),
                    file=SAFE_STDOUT,
                )
            )


    transcription_path = Path(preprocess_text_config.transcription_path)
    cleaned_path = Path(preprocess_text_config.cleaned_path) if preprocess_text_config.cleaned_path else None
    train_path = Path(preprocess_text_config.train_path)
    val_path = Path(preprocess_text_config.val_path)
    config_path = Path(preprocess_text_config.config_path)
    val_per_lang = int(preprocess_text_config.val_per_lang)
    max_val_total = int(preprocess_text_config.max_val_total)
    use_jp_extra = hps.version.endswith("JP-Extra")

    preprocess(
    transcription_path=transcription_path,
    cleaned_path=cleaned_path,
    train_path=train_path,
    val_path=val_path,
    config_path=config_path,
    val_per_lang=val_per_lang,
    max_val_total=max_val_total,
    use_jp_extra=use_jp_extra,
    yomi_error="use",
    correct_path=True,
    train_list_clean_parentheses=True,
    )

    logger.info(f"bert.pt is generated! total: {len(lines)} bert.pt files.")
