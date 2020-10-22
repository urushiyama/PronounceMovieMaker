from janome.tokenizer import Tokenizer
import jaconv
import soundfile as sf
import librosa
import cv2
import numpy as np
from PIL import Image
import PySimpleGUI as sg
import ffmpeg
from PySegmentKit import PySegmentKit, PSKError

import os
import sys
import subprocess
from pathlib import Path
import shutil
import random
from enum import Enum, auto

def subprocess_args(include_stdout=True):
    if hasattr(subprocess, 'STARTUPINFO'):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        env = os.environ
    else:
        si = None
        env = None
 
    if include_stdout:
        ret = {'stdout': subprocess.PIPE}
    else:
        ret = {}
 
    ret.update({'stdin': subprocess.DEVNULL,
                'stderr': subprocess.DEVNULL,
                'startupinfo': si,
                'env': env })
    return ret

tokenizer = Tokenizer(mmap=False)

data_path = Path('./data/')
# 音声ファイル（wavファイル）とスクリプト（txtファイル）を入れておくフォルダ
def scripts_path():
    return data_path.joinpath('scripts/')
# 口パク動画生成用の画像を入れておくフォルダ
def images_path():
    return data_path.joinpath('images/')

# ファイル出力先フォルダ
target_path = Path('./target/')

# 元画像データのキャッシュ
images = {}
# 出力の幅と高さ、最大の元画像データの大きさと一致する大きさでの出力となる
width, height = (0, 0)

# 口パク動画のFPS
fps = 60

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
        for imagefile in images_path().glob("**/*.{}".format(extension)):
            key = imagefile.parent.name
            image_array = np.fromfile(str(imagefile), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
            if image is None: sys.exit('Error: cannot open image: {}'.format(str(imagefile)))
            # Tweak for format image into BGRA. Alpha is retained.
            image = overlayImage(image, image, (0, 0))
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

def generate_transcripts():
    for textfile in scripts_path().glob('*.txt'):
        stem = textfile.stem

        wavfile = scripts_path().joinpath("{}.wav".format(stem))
        if not wavfile.exists(): continue
        target_wavfile = target_path.joinpath("{}.wav".format(stem))
        format_sound(wavfile, target_wavfile)

        with textfile.open(encoding="utf-8") as f:
            # Juliusに投げる発音のテキストデータを作成する
            pronounce = ''.join([token.surface if token.phonetic == '*' else token.phonetic for token in tokenizer.tokenize(jaconv.normalize(f.readline()))])
            pronounce = jaconv.kata2hira(pronounce)
            pronounce = pronounce.replace('。', ' sp ').replace('、', ' sp ').replace('・', ' sp ')
            pronounce = pronounce.replace('「', '').replace('」', '')
            pronounce = pronounce.replace('ゕ', 'か').replace('ゖ', 'か')
            target_textfile = target_path.joinpath("{}.txt".format(stem))
            target_textfile.write_text(pronounce, encoding="utf-8")

def segment_transcripts():
    sk = PySegmentKit(str(target_path))
    try:
        segmented = sk.segment()
        return segmented
    except PSKError as e:
        sg.popup_error('音素セグメンテーションができませんでした：{}'.format(e))
        return None

def generate_movies(segmented):
    for result in segmented.keys():
        stem = Path(result).stem
        target_wavfile = target_path.joinpath("{}.wav".format(stem))

        # 口パク動画のセットアップ
        target_videofile = target_path.joinpath("{}.mov".format(stem))

        # 各フレームの画像を出力する
        frames_path = target_path.joinpath('{}-frames'.format(stem))
        if frames_path.exists():
            shutil.rmtree(frames_path, ignore_errors=True)
        frames_path.mkdir()

        frame = 0
        undetermined = 0
        previous_phoneme = 'silent'
        for begin, end, phoneme in segmented[result]:
            # フォルダ名に合わせてリプレース
            phoneme = phoneme.replace(':', '-').replace('N', 'nn')
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
                        cv2.imwrite(str(frames_path.joinpath('{:07d}.png'.format(frame - undetermined))), image)
                        undetermined -= 1
                    # 現在のフレームをphonemeに対応する母音の画像で埋める
                    cv2.imwrite(str(frames_path.joinpath('{:07d}.png'.format(frame))), image)

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
                        cv2.imwrite(str(frames_path.joinpath('{:07d}.png'.format(frame - undetermined))), image)
                        undetermined -= 1
                    # 現在のフレームをphonemeに対応する母音の画像で埋める
                    cv2.imwrite(str(frames_path.joinpath('{:07d}.png'.format(frame))), image)

                elif phoneme in {'q'}:
                    try:
                        image = images[phoneme]
                    except KeyError:
                        try:
                            # 促音の場合、それ以前の画像を引き伸ばすことを試みる
                            image = images[previous_phoneme]
                        except KeyError:
                            image = images['fallback']
                    cv2.imwrite(str(frames_path.joinpath('{:07d}.png'.format(frame))), image)
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
                        cv2.imwrite(str(frames_path.joinpath('{:07d}.png'.format(frame - undetermined))), image)
                        undetermined -= 1
                    cv2.imwrite(str(frames_path.joinpath('{:07d}.png'.format(frame))), image)
                
                else:
                    # 上記以外の子音の処理
                    try:
                        image = images[phoneme]
                        # undetermined分のフレームを現在のフレームの画像で埋める
                        while undetermined > 0:
                            cv2.imwrite(str(frames_path.joinpath('{:07d}.png'.format(frame - undetermined))), image)
                            undetermined -= 1
                        cv2.imwrite(str(frames_path.joinpath('{:07d}.png'.format(frame))), image)
                    except KeyError:
                        undetermined += 1
                
                print("frame {frame}: {phoneme}".format(frame = frame, phoneme = phoneme))
                frame += 1
                previous_phoneme = phoneme
                continue
        
        if target_videofile.exists():
            target_videofile.unlink()
        (
            ffmpeg
            .input(str(frames_path) + '/%07d.png', framerate=fps)
            .output(str(target_videofile), vcodec='prores', pix_fmt='yuva444p10le')
            .run()
        )

        # 出力した動画ファイルにもとの音声ファイルを埋め込んだファイルも出力する
        target_avfile = target_path.joinpath('{}-with-audio.mov'.format(stem))
        if target_avfile.exists():
            target_avfile.unlink()
        input_video = ffmpeg.input(str(target_videofile))
        input_audio = ffmpeg.input(str(target_wavfile))
        (
            ffmpeg
            .concat(input_video, input_audio, v=1, a=1)
            .output(str(target_avfile), vcodec='prores', pix_fmt='yuva444p10le')
            .run()
        )

def format_sound(source_file, target_file):
    # wavファイルのサンプリングレートを44.1 kHzから16 kHzに変換する
    _convert_sound_format(source_file, target_file, sample_rate=16000)
    # juliusが0出力を除去することによる時間のずれを防ぐためのランダムノイズを追加する
    _add_random_noise_to_sound(target_file, target_file, ignore_non_zero=True)

def _convert_sound_format(source_file, target_file, sample_rate=None, subtype="PCM_16"):
    data, new_sample_rate = librosa.load(str(source_file), sr=sample_rate)
    sf.write(str(target_file), data, new_sample_rate, subtype=subtype)

def _add_random_noise_to_sound(source_file: Path, target_file: Path, subtype="PCM_16", noise_amp=5e-07, ignore_non_zero=False):
    data, sample_rate = librosa.load(str(source_file), sr=None)
    if ignore_non_zero:
        data = list(map(lambda x: x + noise_amp * 2 * random.random() - noise_amp if x == 0 else x, data))
    else:
        data = list(map(lambda x: x + noise_amp * 2 * random.random() - noise_amp, data))
    sf.write(str(target_file), data, sample_rate, subtype=subtype)

sg.theme('BlueMono')

class WindowElementKey(Enum):
    FORMAT_BUTTON = auto()
    FORMAT_CHECKBOX = auto()
    GENERATE_TRANSCRIPT_BUTTON = auto()
    GENERATE_MOVIE_BUTTON = auto()
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
    [sg.Text('5. よみがなのテキストファイルを生成')],
    [sg.Button('よみがな生成', key=WindowElementKey.GENERATE_TRANSCRIPT_BUTTON, pad=((15, 5), 3))],
    [sg.Text('6. （必要であれば）よみがなを修正')],
    [sg.Text('※ひらがなと無声区間" sp "のみ入力可能です', font='Courier 9', pad=((15, 5), 3))],
    [sg.Text('7. 口パク動画を生成')],
    [sg.Button('動画生成', key=WindowElementKey.GENERATE_MOVIE_BUTTON, pad=((15, 5), 3))],
    [sg.Output(pad=((15, 5), 3), echo_stdout_stderr=True, key=WindowElementKey.OUTPUT)],
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
    if not (scripts_path().exists() and scripts_path().is_dir()):
        return False
    if not (images_path().exists() and images_path().is_dir()):
        return False
    images_children = set()
    for child in images_path().iterdir():
        images_children.add(child.name)
    return required_data_directories.issubset(images_children)

def formatDataDirectory():
    try:
        scripts_path().mkdir(exist_ok=True)
        images_path().mkdir(exist_ok=True)
    except FileExistsError:
        sg.popup_error('データフォルダには次の名前のファイルを入れないでください：\r\nscripts, images')
        return False
    try:
        for required_directory in required_data_directories:
            images_path().joinpath(required_directory).mkdir(exist_ok=True)
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
        main_window[WindowElementKey.GENERATE_TRANSCRIPT_BUTTON].update(disabled=False)
        main_window[WindowElementKey.GENERATE_MOVIE_BUTTON].update(disabled=False)
    else:
        main_window[WindowElementKey.GENERATE_TRANSCRIPT_BUTTON].update(disabled=True)
        main_window[WindowElementKey.GENERATE_MOVIE_BUTTON].update(disabled=True)
    
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
    if event == WindowElementKey.GENERATE_TRANSCRIPT_BUTTON:
        main_window[WindowElementKey.GENERATE_MOVIE_BUTTON].update(disabled=True)
        generate_transcripts()
        main_window[WindowElementKey.GENERATE_MOVIE_BUTTON].update(disabled=False)
    if event == WindowElementKey.GENERATE_MOVIE_BUTTON:
        main_window[WindowElementKey.OUTPUT].update(value='')
        main_window[WindowElementKey.GENERATE_MOVIE_BUTTON].update(disabled=True)

        main_window[WindowElementKey.GENERATE_MOVIE_BUTTON].update(disabled=True)
        segmented = segment_transcripts()
        main_window[WindowElementKey.GENERATE_MOVIE_BUTTON].update(disabled=False)

        if segmented == None:
            continue

        loadImages()
        # fallbackに画像がない場合には処理を停止する
        if not 'fallback' in images:
            sg.popup_error('imagesフォルダ下のfallbackフォルダには元データとなる画像を入れてください')
            continue
        resizeImages()
        overlayImagesOnBackground()
        generate_movies(segmented)
