#!/usr/bin/env python3
from enum import Enum, auto
from pathlib import Path
import random
import sys
import shutil

from janome.tokenizer import Tokenizer
import jaconv
import soundfile as sf
import librosa
import cv2
import numpy as np
from PIL import Image
import ffmpeg
from PySegmentKit import PySegmentKit, PSKError
from pathvalidate import ValidationError, validate_filepath

from kivy.config import Config
Config.set('graphics', 'width', 400)
Config.set('graphics', 'height', 520)
Config.set('graphics', 'resizable', False)
import kivy
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.core.window import Window
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.resources import resource_add_path

resource_add_path("./resources")
LabelBase.register(DEFAULT_FONT, "SourceHanSans-Regular.ttc")

class Error(Exception):
    """ the base class of this application errors.

    Attributes:
        message -- short message which describes the error.
        description -- long description of the error.
    """
    def __init__(self, message, description):
        self.message = message
        self.description = description

    def __str__(self):
        return self.message

class LoadImageError(Error):
    """ raised when image loading is failed.
    """
    pass

class SegmentationError(Error):
    """ raised when phoneme segmentation is failed.
    """
    pass

class FormatDataDirectoryError(Error):
    """ raised when data directory formatting is failed.
    """
    pass

class AccessTargetDirectoryError(Error):
    """ raised when target directory access is failed.
    """

tokenizer = Tokenizer(mmap=False)

data_path = Path('')
# 音声ファイル（wavファイル）とスクリプト（txtファイル）を入れておくフォルダ
def scripts_path():
    return data_path.joinpath('scripts/')
# 口パク動画生成用の画像を入れておくフォルダ
def images_path():
    return data_path.joinpath('images/')

# ファイル出力先フォルダ
target_path = Path('')

# 元画像データのキャッシュ
images = {}
# 出力の幅と高さ、最大の元画像データの大きさと一致する大きさでの出力となる
width, height = (0, 0)

# 口パク動画のFPS
fps = 60

def overlay_image(background, overlay, location):
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

def load_images():
    global height, width
    load_error_description = ""
    load_error_images_path = images_path().joinpath('load_errors/')
    if load_error_images_path.exists():
        shutil.rmtree(load_error_images_path, ignore_errors=True)
    for extension in ('tiff', 'jpeg', 'jpg', 'png'):
        for imagefile in images_path().glob("**/*.{}".format(extension)):
            key = imagefile.parent.name
            image_array = np.fromfile(str(imagefile), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
            if image is None:
                load_error_description += '- {}\n'.format(str(imagefile))
                if not load_error_images_path.exists():
                    load_error_images_path.mkdir()
                shutil.move(str(imagefile), str(load_error_images_path))
                continue
            # Tweak for format image into BGRA. Alpha is retained.
            image = overlay_image(image, image, (0, 0))
            images[key] = image
            h, w = image.shape[:2]
            if h > height: height = h
            if w > width: width = w
    if load_error_description != "":
        load_error_description += '開けなかった画像は次のフォルダに移動しました：{}\nこれらの画像は口パク動画の生成には使用されません。'.format(str(load_error_images_path))
        raise LoadImageError('画像を開けませんでした', load_error_description)

def resize_images():
    # 最大の画像の縦横サイズに縮尺を合わせる
    for k in images.keys():
        images[k] = cv2.resize(images[k], (width, height))

def overlay_images_on_background():
    # 背景画像がある場合には組み合わせておく
    if 'background' in images:
        for k in images.keys():
            if k == 'background': continue
            images[k] = overlay_image(images['background'], images[k], (0, 0))

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
        raise SegmentationError('音素セグメンテーションができませんでした', str(e))

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

class FormatCheckerStatus(Enum):
    UNSPECIFIED = 'データフォルダ未指定'
    UNFORMATTED = '未フォーマット'
    FORMATTED = 'フォーマット済み'
    
    def __bool__(self):
        return self == FormatCheckerStatus.FORMATTED

class FFMPEGCheckerStatus(Enum):
    NOT_INSTALLED = '※ffmpegのインストールが必要です'
    INSTALLED = '※ffmpegのインストールを確認できました'

    def __bool__(self):
        return self == FFMPEGCheckerStatus.INSTALLED
    
    @classmethod
    def from_bool(cls, installed: bool):
        if installed:
            return FFMPEGCheckerStatus.INSTALLED
        else:
            return FFMPEGCheckerStatus.NOT_INSTALLED

required_data_directories = frozenset({
    'a', 'a-', 'i', 'i-', 'u', 'u-', 'e', 'e-', 'o', 'o-',
    'b', 'by', 'ch', 'd', 'dy', 'f', 'g', 'gy', 'h', 'hy',
    'j', 'k', 'ky', 'm', 'my', 'n', 'ny', 'p', 'py', 'r', 'ry',
    's', 'sh', 't', 'ts', 'ty', 'w', 'y', 'z', 'zy',
    'nn', 'q', 'fallback', 'silent', 'background'})

def path_is_valid(path):
    try:
        validate_filepath(str(path), platform='auto')
    except ValidationError as e:
        return False
    return str(path) != '.'

def data_directory_is_formatted():
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

def format_data_directory():
    try:
        scripts_path().mkdir(exist_ok=True)
        images_path().mkdir(exist_ok=True)
    except FileExistsError:
        raise FormatDataDirectoryError('データフォルダのフォーマットができませんでした', 'データフォルダには次の名前のファイルを入れないでください：\nscripts, images')
    try:
        for required_directory in required_data_directories:
            images_path().joinpath(required_directory).mkdir(exist_ok=True)
    except FileExistsError:
        raise FormatDataDirectoryError('データフォルダのフォーマットができませんでした', 'データフォルダ下のimagesフォルダには次の名前のファイルを入れないでください：\n{}'.format(', '.join(required_data_directories)))

def target_directory_is_formatted():
    if not path_is_valid(target_path):
        return False
    if not (target_path.exists() and target_path.is_dir()):
        return False
    if ' ' in target_path.name:
        return False
    return True

def ffmpeg_is_installed():
    return shutil.which('ffmpeg') is not None

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def is_dir(self, dirname, filename):
        return Path(dirname).joinpath(filename).is_dir()

class ErrorDialog(FloatLayout):
    description = ObjectProperty(None)
    cancel = ObjectProperty(None)

class Root(FloatLayout):
    datafile_input = ObjectProperty(None)
    targetfile_input = ObjectProperty(None)
    ffmpeg_checker_label = ObjectProperty(Label())
    format_button = ObjectProperty(Button())
    format_checkbox = ObjectProperty(CheckBox())
    format_label = ObjectProperty(Label())
    transcript_button = ObjectProperty(Button())
    movie_button = ObjectProperty(Button())

    def textinput_focus(self, target, value):
        global data_path, target_path
        if not value:
            data_path = Path(self.datafile_input.text)
            target_path = Path(self.targetfile_input.text)
            self.update_view()

    def dismiss_popup(self):
        self.update_view()
        self._popup.dismiss()
    
    def show_data_load(self):
        content = LoadDialog(load=self.data_load, cancel=self.dismiss_popup)
        self._popup = Popup(title="データフォルダを選択してください", content=content, size_hint=(0.9, 0.9))
        self._popup.open()
    
    def data_load(self, path, filename):
        global data_path
        self.datafile_input.text = path
        data_path = Path(path)
        self.dismiss_popup()
    
    def show_target_load(self):
        content = LoadDialog(load=self.target_load, cancel=self.dismiss_popup)
        self._popup = Popup(title="出力先フォルダを選択してください", content=content, size_hint=(0.9, 0.9))
        self._popup.open()
    
    def target_load(self, path, filename):
        global target_path
        self.targetfile_input.text = path
        target_path = Path(path)
        self.dismiss_popup()

    def format_data_directory_pressed(self):
        self.format_button.disabled = True
        try:
            if not (path_is_valid(data_path) and data_path.exists()):
                raise FormatDataDirectoryError('データフォルダのフォーマットができませんでした', '指定したデータフォルダのパスを確認してください')
            format_data_directory()
        except Error as e:
            content = ErrorDialog(description=e.description, cancel=self.dismiss_popup)
            self._popup = Popup(title=e.message, content=content, size_hint=(0.9, 0.9))
            self._popup.open()
        self.update_view()

    def generate_transcripts_pressed(self):
        self.movie_button.disabled = True
        try:
            if not target_directory_is_formatted():
                raise AccessTargetDirectoryError('出力先フォルダにアクセスできませんでした', '出力先フォルダには既に存在するフォルダ名を指定してください')
            generate_transcripts()
        except Error as e:
            content = ErrorDialog(description=e.description, cancel=self.dismiss_popup)
            self._popup = Popup(title=e.message, content=content, size_hint=(0.9, 0.9))
            self._popup.open()
        self.movie_button.disabled = False

    def generate_movies_pressed(self):
        self.movie_button.disabled = True
        self.transcript_button.disabled = True
        try:
            if not target_directory_is_formatted():
                raise AccessTargetDirectoryError('出力先フォルダにアクセスできませんでした', '出力先フォルダには既に存在するフォルダ名を指定してください')
            segmented = segment_transcripts()
            self.transcript_button.disabled = False
            load_images()
            # fallbackに画像がない場合には処理を停止する
            if not 'fallback' in images:
                raise LoadImageError('必要な画像が不足しています', 'imagesフォルダ下のfallbackフォルダには元データとなる画像を入れてください')
            resize_images()
            overlay_images_on_background()
            generate_movies(segmented)
        except Error as e:
            content = ErrorDialog(description=e.description, cancel=self.dismiss_popup)
            self._popup = Popup(title=e.message, content=content, size_hint=(0.9, 0.9))
            self._popup.open()
        self.transcript_button.disabled = False
        self.movie_button.disabled = False
    
    def update_view(self):
        self.ffmpeg_checker_label.text = FFMPEGCheckerStatus.from_bool(ffmpeg_is_installed()).value
        if path_is_valid(data_path) and data_directory_is_formatted():
            self.format_button.disabled = True
            self.format_checkbox.active = bool(FormatCheckerStatus.FORMATTED)
            self.format_label.text = FormatCheckerStatus.FORMATTED.value
            if target_directory_is_formatted():
                self.transcript_button.disabled = False
                self.movie_button.disabled = not ffmpeg_is_installed()
            else:
                self.transcript_button.disabled = True
                self.movie_button.disabled = True  
        else:
            self.format_button.disabled = False
            self.format_checkbox.active = bool(FormatCheckerStatus.UNFORMATTED)
            self.format_label.text = FormatCheckerStatus.UNFORMATTED.value
            if not (path_is_valid(data_path) and data_path.exists()):
                self.format_button.disabled = True
                self.format_label.text = FormatCheckerStatus.UNSPECIFIED.value
            self.transcript_button.disabled = True
            self.movie_button.disabled = True

class Main(App):
    def on_start(self):
        self.root.update_view()

Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('ErrorDialog', cls=ErrorDialog)

if __name__ == '__main__':
    Main().run()
