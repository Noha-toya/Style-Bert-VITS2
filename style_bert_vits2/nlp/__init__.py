from typing import TYPE_CHECKING, Optional

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp.symbols import (
    LANGUAGE_ID_MAP,
    LANGUAGE_TONE_START_MAP,
    SYMBOLS,
)


# __init__.py は配下のモジュールをインポートした時点で実行される
# PyTorch のインポートは重いので、型チェック時以外はインポートしない
if TYPE_CHECKING:
    import torch


__symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    language: Languages,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    removed_parentheses_text: Optional[str] = None,
    removed_parentheses_word2ph: Optional[list[int]] = None,
    removed_parentheses_fags: Optional[list[bool]] = None,
) -> "torch.Tensor":
    """
    テキストから BERT の特徴量を抽出する

    Args:
        text (str): テキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        language (Languages): テキストの言語
        device (str): 推論に利用するデバイス
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        torch.Tensor: BERT の特徴量
    """

    if language == Languages.JP:
        from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature
        return extract_bert_feature(text, word2ph, device, assist_text, assist_text_weight,removed_parentheses_text,removed_parentheses_word2ph,removed_parentheses_fags)

    elif language == Languages.EN:
        from style_bert_vits2.nlp.english.bert_feature import extract_bert_feature
    elif language == Languages.ZH:
        from style_bert_vits2.nlp.chinese.bert_feature import extract_bert_feature
    else:
        raise ValueError(f"Language {language} not supported")

    return extract_bert_feature(text, word2ph, device, assist_text, assist_text_weight)


def clean_text(
    text: str,
    language: Languages,
    use_jp_extra: bool = True,
    raise_yomi_error: bool = False,
) -> tuple[str, list[str], list[int], list[int], str, list[str], list[int], list[int], list[bool] ]:
    """
    テキストをクリーニングし、音素に変換する

    Args:
        text (str): クリーニングするテキスト
        language (Languages): テキストの言語
        use_jp_extra (bool, optional): テキストが日本語の場合に JP-Extra モデルを利用するかどうか。Defaults to True.
        raise_yomi_error (bool, optional): False の場合、読めない文字が消えたような扱いとして処理される。Defaults to False.

    Returns:
        tuple[str, list[str], list[int], list[int], str, list[str], list[int], list[int], list[int] ]: クリーニングされたテキストと、音素・アクセント・元のテキストの各文字に音素が何個割り当てられるかのリスト
    """

    # Changed to import inside if condition to avoid unnecessary import
    if language == Languages.JP:
        from style_bert_vits2.nlp.japanese.g2p import g2p
        from style_bert_vits2.nlp.japanese.normalizer import normalize_text
        
        non_rm_norm_text = normalize_text(text)
        rm_par_text,rm_par_fags = remove_parentheses_sentences(non_rm_norm_text)          # for mood
        non_rm_phones, non_rm_tones, non_rm_word2ph = g2p(non_rm_norm_text, use_jp_extra, raise_yomi_error)
        rm_par_phones, rm_par_tones, rm_par_word2ph = g2p(rm_par_text, use_jp_extra, raise_yomi_error) # for mood

    elif language == Languages.EN:
        from style_bert_vits2.nlp.english.g2p import g2p
        from style_bert_vits2.nlp.english.normalizer import normalize_text

        non_rm_norm_text = normalize_text(text)
        rm_par_text,rm_par_fags = remove_parentheses_sentences(non_rm_norm_text)          # for mood
        non_rm_phones, non_rm_tones, non_rm_word2ph = g2p(non_rm_norm_text)
        rm_par_phones, rm_par_tones, rm_par_word2ph = g2p(rm_par_text) # for mood
    elif language == Languages.ZH:
        from style_bert_vits2.nlp.chinese.g2p import g2p
        from style_bert_vits2.nlp.chinese.normalizer import normalize_text

        non_rm_norm_text = normalize_text(text)
        rm_par_text,rm_par_fags = remove_parentheses_sentences(non_rm_norm_text)          # for mood
        non_rm_phones, non_rm_tones, non_rm_word2ph = g2p(non_rm_norm_text)
        rm_par_phones, rm_par_tones, rm_par_word2ph = g2p(rm_par_text) # for mood    else:
        raise ValueError(f"Language {language} not supported")

    return  rm_par_text, rm_par_phones, rm_par_tones, rm_par_word2ph,non_rm_norm_text, non_rm_phones, non_rm_tones, non_rm_word2ph, rm_par_fags


def cleaned_text_to_sequence(
    cleaned_phones: list[str], tones: list[int], language: Languages
) -> tuple[list[int], list[int], list[int]]:
    """
    音素リスト・アクセントリスト・言語を、テキスト内の対応する ID に変換する

    Args:
        cleaned_phones (list[str]): clean_text() でクリーニングされた音素のリスト
        tones (list[int]): 各音素のアクセント
        language (Languages): テキストの言語

    Returns:
        tuple[list[int], list[int], list[int]]: List of integers corresponding to the symbols in the text
    """

    phones = [__symbol_to_id[symbol] for symbol in cleaned_phones]
    tone_start = LANGUAGE_TONE_START_MAP[language]
    tones = [i + tone_start for i in tones]
    lang_id = LANGUAGE_ID_MAP[language]
    lang_ids = [lang_id for i in phones]

    return phones, tones, lang_ids



def remove_parentheses_sentences(text):
    """
    mood のため、カッコの文章の削除と、カッコの文章の文字についてフラグ作成

    Args:
        s (str): 日本語のテキスト

    Returns:
        filtered_result: カッコの文章の削除後のテキスト
        flags: 削除した文字にフラグ
    """
    open_count = text.count('(')
    close_count = text.count(')')
    
    # バランスを取る
    if open_count > close_count:
        diff = open_count - close_count
        ignored_open = 0
        new_text = ""
        for char in text:
            if char == '(' and ignored_open < diff:
                ignored_open += 1
            else:
                new_text += char
        text = new_text
    elif close_count > open_count:
        diff = close_count - open_count
        ignored_close = 0
        new_text = ""
        for char in reversed(text):
            if char == ')' and ignored_close < diff:
                ignored_close += 1
            else:
                new_text = char + new_text
        text = new_text
    
    # カッコとその中身を削除
    result_text = ""
    flags = []
    skip = 0
    
    for i, char in enumerate(text):
        if char == '(':
            skip += 1
            flags.append(False)
        elif char == ')' and skip > 0:
            skip -= 1
            flags.append(False)
        elif skip > 0:
            flags.append(False)
        else:
            result_text += char
            flags.append(True)
        
    flags.insert(0,True)
    flags.append(True)
    
    return result_text, flags

