import MeCab
import jaconv
import soundfile as sf
import librosa
import cv2
import numpy as np
from PIL import Image
import moviepy.editor as mp
import PySimpleGUI as sg

import sys
import subprocess
from pathlib import Path
import shutil
import random
from enum import Enum, auto

pronounce_tagger = MeCab.Tagger("--node-format=%pS%f[8] --unk-format=%M --eos-format=\n")

data_path = Path('./data/')
# 音声ファイル（wavファイル）とスクリプト（txtファイル）を入れておくフォルダ
scripts_path = data_path.joinpath('scripts/')
# 口パク動画生成用の画像を入れておくフォルダ
images_path = data_path.joinpath('images/')

# ファイル出力先フォルダ
target_path = Path('./target/')

# 元画像データのキャッシュ
images = {}
# 出力の幅と高さ、最大の元画像データの大きさと一致する大きさでの出力となる
width, height = (0, 0)

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

def loadImages():
    global height, width
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

def resizeImages():
    # 最大の画像の縦横サイズに縮尺を合わせる
    for k in images.keys():
        images[k] = cv2.resize(images[k], (width, height))

def overlayImagesOnBackground():
    # 背景画像がある場合には組み合わせておく
    if 'background' in images:
        for k in images.keys():
            if k == 'background': continue
            images[k] = overlayImage(images['background'], images[k], (0, 0))

def generate():
    for textfile in scripts_path.glob('*.txt'):
        with textfile.open(encoding="utf-8") as f:
            stem = textfile.stem
            wavfile = scripts_path.joinpath("{}.wav".format(stem))
            if not wavfile.exists(): continue

            # wavファイルのサンプリングレートを44.1 kHzから16 kHzに変換する
            data, _ = librosa.load(str(wavfile), sr=16000)
            # ついでにjuliusが0出力を除去することによる時間のずれを防ぐためのランダムノイズを追加する
            data = list(map(lambda x: x + 1e-06 * random.random() - 5e-07 if x == 0 else x, data))
            target_wavfile = target_path.joinpath("{}.wav".format(stem))
            sf.write(str(target_wavfile), data, 16000, subtype="PCM_16")

            # Juliusに投げる発音のテキストデータを作成する
            pronounce = jaconv.kata2hira(pronounce_tagger.parse(f.readline())).replace('。', ' sp ').replace('、', ' sp ').replace('・', ' sp ')
            target_textfile = target_path.joinpath("{}.txt".format(stem))
            target_textfile.write_text(pronounce, encoding="utf-8")

            # segment_julius.plを用いて音素セグメンテーションを行う
            subprocess.run(['perl', './segment_julius.pl', str(target_path.resolve())], cwd='./segmentation-kit')

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
                                # undetermined分のフレームを現在のフレームの画像で埋める
                                while undetermined > 0:
                                    video.write(image)
                                    undetermined -= 1
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

sg.theme('BlueMono')

class WindowElementKey(Enum):
    FORMAT_BUTTON = auto()
    FORMAT_CHECKBOX = auto()
    GENERATE_BUTTON = auto()
    OUTPUT = auto()

class FormatCheckerStatus(Enum):
    UNFORMATTED = '未フォーマット'
    FORMATTED = 'フォーマット済み'
    
    def __bool__(self):
        return self == FormatCheckerStatus.FORMATTED

main_window = sg.Window('Pronounce Movie Maker', [
    [sg.Text('1. データフォルダを選択')],
    [sg.InputText(default_text=str(data_path.resolve()), enable_events=True, pad=((15, 5), 3)), sg.FolderBrowse(initial_folder=str(data_path.resolve()), enable_events=True)],
    [sg.Text('2. データフォルダをフォーマット')],
    [sg.Button('フォーマット', key=WindowElementKey.FORMAT_BUTTON, pad=((15, 5), 3)), sg.Checkbox(FormatCheckerStatus.UNFORMATTED.value, disabled=True, key=WindowElementKey.FORMAT_CHECKBOX) ],
    [sg.Text('3. データフォルダ内部に画像を配置（png, jpg, tiff）')],
    [sg.Text('4. 出力先フォルダを選択')],
    [sg.Text('※フォルダ名に半角スペースを含めないでください', font='Courier 9', pad=((15, 5), 3))],
    [sg.InputText(default_text=str(target_path.resolve()), enable_events=True, pad=((15, 5), 3)), sg.FolderBrowse(initial_folder=str(target_path.resolve()), enable_events=True)],
    [sg.Text('5. 口パク動画を生成')],
    [sg.Button('生成', key=WindowElementKey.GENERATE_BUTTON, pad=((15, 5), 3))],
    [sg.Output(pad=((15, 5), 3), key=WindowElementKey.OUTPUT)],
    [sg.HorizontalSeparator()],
    [sg.Exit()]], finalize=True)

required_data_directories = frozenset({
    'a', 'a-', 'i', 'i-', 'u', 'u-', 'e', 'e-', 'o', 'o-',
    'b', 'by', 'ch', 'd', 'dy', 'f', 'g', 'gy', 'h', 'hy',
    'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'p', 'py', 'r', 'ry',
    's', 'sh', 't', 'ts', 'ty', 'w', 'y', 'z', 'zy',
    'nn', 'q', 'fallback', 'silent', 'background'})

def dataDirectoryIsFormatted():
    if not (data_path.exists() and data_path.is_dir()):
        return False
    scripts_path = data_path.joinpath('scripts/')
    images_path = data_path.joinpath('images/')
    if not (scripts_path.exists() and scripts_path.is_dir()):
        return False
    if not (images_path.exists() and images_path.is_dir()):
        return False
    images_children = set()
    for child in images_path.iterdir():
        images_children.add(child.name)
    return required_data_directories.issubset(images_children)

def formatDataDirectory():
    scripts_path = data_path.joinpath('scripts/')
    images_path = data_path.joinpath('images/')
    try:
        scripts_path.mkdir(exist_ok=True)
        images_path.mkdir(exist_ok=True)
    except FileExistsError:
        sg.popup_error('データフォルダには次の名前のファイルを入れないでください：\r\nscripts, images')
        return False
    try:
        for required_directory in required_data_directories:
            images_path.joinpath(required_directory).mkdir(exist_ok=True)
    except FileExistsError:
        sg.popup_error('データフォルダ下のimagesフォルダには次の名前のファイルを入れないでください：\r\n{}'.format(', '.join(required_data_directories)))
        return False
    return True

def targetDirectoryIsFormatted():
    if not (target_path.exists() and target_path.is_dir()):
        return False
    if ' ' in target_path.name:
        return False
    return True

while True:
    if dataDirectoryIsFormatted():
        main_window[WindowElementKey.FORMAT_BUTTON].update(disabled=True)
        main_window[WindowElementKey.FORMAT_CHECKBOX].update(value=bool(FormatCheckerStatus.FORMATTED), text=FormatCheckerStatus.FORMATTED.value)
    else:
        main_window[WindowElementKey.FORMAT_BUTTON].update(disabled=False)
        main_window[WindowElementKey.FORMAT_CHECKBOX].update(value=bool(FormatCheckerStatus.UNFORMATTED), text=FormatCheckerStatus.UNFORMATTED.value)
    if targetDirectoryIsFormatted():
        main_window[WindowElementKey.GENERATE_BUTTON].update(disabled=False)
    else:
        main_window[WindowElementKey.GENERATE_BUTTON].update(disabled=True)
    
    event, values = main_window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':
        main_window.close()
        sys.exit(0)

    data_path = Path(values[0])
    target_path = Path(values[1])

    if event == WindowElementKey.FORMAT_BUTTON:
        if not (data_path.exists() and data_path.is_dir()):
            sg.popup_error('データフォルダにはファイルシステム上に存在するフォルダを指定してください')
            continue
        formatDataDirectory()
    if event == WindowElementKey.GENERATE_BUTTON:
        if not (target_path.exists() and target_path.is_dir()):
            sg.popup_error('出力先フォルダにはファイルシステム上に存在するフォルダを指定してください')
            continue
        if ' ' in target_path.name:
            sg.popup_error('出力先フォルダ名には半角スペースを含めないでください')
            continue
        
        main_window[WindowElementKey.OUTPUT].update(value='')
        main_window[WindowElementKey.GENERATE_BUTTON].update(disabled=True)

        loadImages()
        # fallbackに画像がない場合には処理を停止する
        if not 'fallback' in images:
            sg.popup_error('imagesフォルダ下のfallbackフォルダには元データとなる画像を入れてください')
            continue
        resizeImages()
        overlayImagesOnBackground()
        generate()
