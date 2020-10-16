import MeCab
import jaconv
import soundfile as sf
import librosa
import cv2
import numpy as np
from PIL import Image
import moviepy.editor as mp

import sys
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
fps = 120


def overlayImage(background, overlay, location):
    background = cv2.cvtColor(background, cv2.COLOR_BGRA2RGBA)
    pil_background = Image.fromarray(np.uint8(background))
    pil_background = pil_background.convert('RGBA')

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
    pil_overlay = Image.fromarray(np.uint8(overlay))
    pil_overlay = pil_overlay.convert('RGBA')

    pil_temp = Image.new('RGBA', pil_background.size, (255, 255, 255, 0))
    pil_temp.paste(pil_overlay, location, pil_overlay)
    result_image = Image.alpha_composite(pil_background, pil_temp)

    return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

images = {}
width, height = (0, 0)
for extension in ('tiff', 'jpeg', 'jpg', 'png'):
    for imagefile in images_path.glob("**/*.{}".format(extension)):
        key = imagefile.parent.name
        image_array = np.fromfile(str(imagefile), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        if image is None: sys.exit('Error: cannot open image: {}'.format(str(imagefile)))
        images[key] = image
        h, w = image.shape[:2]
        if h > height: height = h
        if w > width: width = w

# fallbackに画像がない場合には処理を停止する
if not 'fallback' in images: sys.exit("Error: no fallback image provided")

# 最大の画像の縦横サイズに縮尺を合わせる
for k in images.keys():
    images[k] = cv2.resize(images[k], (width, height))
    h, w = images[k].shape[:2]

# 背景画像がある場合には組み合わせておく
if 'background' in images:
    for k in images.keys():
        if k == 'background': continue
        images[k] = overlayImage(images['background'], images[k], (0, 0))

# ファイル出力先フォルダ
target_path = Path('./target/')
if not target_path.exists(): target_path.mkdir()

for textfile in scripts_path.glob('*.txt'):
    with textfile.open(encoding="utf-8") as f:
        stem = textfile.stem
        wavfile = scripts_path.joinpath("{}.wav".format(stem))
        if not wavfile.exists(): continue

        # wavファイルのサンプリングレートを44.1 kHzから16 kHzに変換する
        data, samplerate = librosa.load(str(wavfile), sr=16000)
        target_wavfile = target_path.joinpath("{}.wav".format(stem))
        sf.write(str(target_wavfile), data, 16000)

        # Juliusに投げる発音のテキストデータを作成する
        pronounce = jaconv.kata2hira(pronounce_tagger.parse(f.readline())).replace('。', ' sp ').replace('、', ' sp ')
        target_textfile = target_path.joinpath("{}.txt".format(stem))
        target_textfile.write_text(pronounce, encoding="utf-8")

        # segment_julius.plを用いて音素セグメンテーションを行う
        subprocess.run(['perl', './segment_julius.pl', '../target'], cwd='./segmentation-kit')

        # 音素セグメンテーションの結果からファイルパスとして使用できない[:]を[-]に、大文字小文字の区別不要のため[N]を[nn]に変換する
        labfile = target_path.joinpath("{}.lab".format(stem))
        if not labfile.exists(): continue
        escaped = labfile.read_text().replace(':', '-').replace('N', 'nn')
        labfile.write_text(escaped)

        # 口パク動画のセットアップ
        # mp4エンコーダ
        fourcc = cv2.VideoWriter_fourcc(*'hev1')
        # 出力する動画
        target_videofile = target_path.joinpath("{}.mp4".format(stem))
        video = cv2.VideoWriter(str(target_videofile), fourcc, fps, (width, height))
        if not video.isOpened(): sys.exit("Error: cannot open video output: {}".format(str(target_textfile)))
        video.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        frame = 0
        undetermined = 0
        previous_phoneme = 'silent'
        with labfile.open() as lf:
            for line in lf:
                begin, end, phoneme = line.split()
                begin = float(begin)
                end = float(end)
                while frame / fps <= end:
                    if frame / fps < begin:
                        phoneme = previous_phoneme
                        # try:
                        #     image = images['silent']
                        # except KeyError:
                        #     image = images['fallback']
                        # video.write(image)

                    if phoneme in {'a', 'i', 'u', 'e', 'o', 'nn'}:
                        try:
                            image = images[phoneme]
                        except KeyError:
                            image = images['fallback']
                        # undetermined分のフレームをphonemeに対応する母音の画像で埋める
                        while undetermined > 0:
                            video.write(image)
                            undetermined -= 1
                        # 現在のフレームをphonemeに対応する母音の画像で埋める
                        video.write(image)

                    elif phoneme in {'a-', 'i-', 'u-', 'e-', 'o-'}:
                        try:
                            image = images[phoneme]
                        except KeyError:
                            try:
                                image = images[phoneme[0]]
                            except KeyError:
                                image = images['fallback']
                        # undetermined分のフレームをphonemeに対応する母音の画像で埋める
                        while undetermined > 0:
                            video.write(image)
                            undetermined -= 1
                        # 現在のフレームをphonemeに対応する母音の画像で埋める
                        video.write(image)

                    elif phoneme in {'q'}:
                        try:
                            image = images[phoneme]
                        except KeyError:
                            try:
                                # 促音の場合、それ以前の画像を引き伸ばすことを試みる
                                image = images[previous_phoneme]
                            except KeyError:
                                image = images['fallback']
                        video.write(image)
                        frame += 1
                        continue

                    elif phoneme in {'silB', 'silE', 'sp'}:
                        phoneme = 'silent'
                        try:
                            image = images['silent']
                        except KeyError:
                            image = images['fallback']
                        # undetermined分のフレームを現在のフレームの画像で埋める
                        while undetermined > 0:
                            video.write(image)
                            undetermined -= 1
                        video.write(image)
                    
                    else:
                        # 上記以外の子音の処理
                        try:
                            image = images[phoneme]
                            video.write(image)
                        except KeyError:
                            undetermined += 1
                    
                    print("frame {frame}: {phoneme}".format(frame = frame, phoneme = phoneme))
                    frame += 1
                    previous_phoneme = phoneme
                    continue
                
        video.release()

        # 出力した動画ファイルにもとの音声ファイルを埋め込んだファイルも出力する
        target_avfile = target_path.joinpath('{}-with-audio.mp4'.format(stem))
        videoclip = mp.VideoFileClip(str(target_videofile)).subclip()
        audioclip = mp.AudioFileClip(str(target_wavfile))
        avclip = videoclip.set_audio(audioclip)
        avclip.write_videofile(str(target_avfile))