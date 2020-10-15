import MeCab
import jaconv
import soundfile as sf
import subprocess
from pathlib import Path
import shutil

pronounce_tagger = MeCab.Tagger("--node-format=%pS%f[8] --unk-format=%M --eos-format=\n")

data_path = Path('./data/')
# 音声ファイル（wavファイル）とスクリプト（txtファイル）を入れておくフォルダ
scripts_path = data_path.joinpath('scripts/')
# 口パク動画生成用の画像を入れておくフォルダ
images_path = data_path.joinpath('images/')
# 口パク動画のFPS
fps = 60

# ファイル出力先フォルダ
target_path = Path('./target/')
if not target_path.exists(): target_path.mkdir()

for textfile in scripts_path.glob('*.txt'):
    with textfile.open(encoding="utf-8") as f:
        stem = textfile.stem
        wavfile = scripts_path.joinpath("{}.wav".format(stem))
        if not wavfile.exists(): break

        # wavファイルのサンプリングレートを44.1 kHzから16 kHzに変換する
        data, samplerate = sf.read(str(wavfile))
        sf.write(str(target_path.joinpath("{}.wav".format(stem))), data, 16000)

        # Juliusに投げる発音のテキストデータを作成する
        pronounce = jaconv.kata2hira(pronounce_tagger.parse(f.readline())).replace('。', ' sp ').replace('、', ' sp ')
        target_textfile = target_path.joinpath("{}.txt".format(stem))
        target_textfile.write_text(pronounce, encoding="utf-8")

        # segment_julius.plを用いて音素セグメンテーションを行う
        subprocess.run(['perl', './segment_julius.pl', '../target'], cwd='./segmentation-kit')

        # 音素セグメンテーションの結果からファイルパスとして使用できない[:]を[-]に変換する
        labfile = target_path.joinpath("{}.lab".format(stem))
        if not labfile.exists(): break
        escaped = labfile.read_text().replace(':', '-')
        labfile.write_text(escaped)
