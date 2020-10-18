# Pronounce Movie Maker

## Requirements

- some environment to run python script (tested by [Anaconda](https://www.anaconda.com/products/individual))
- some third-party python libraries (run `pip install -r requirements.txt` to use)
- some environment to run perl script (tested by [ActivePerl](https://www.activestate.com/products/perl/downloads/))
- ffmpeg should be installed and the path to its bin shoubld be created

## Directory Structure

```
PronounceMovieMaker
├─data
│  ├─images
│  │  ├─a
│  │  ├─a-
│  │  ├─b
│  │  ├─background
│  │  ├─by
│  │  ├─ch
│  │  ├─d
│  │  ├─dy
│  │  ├─e
│  │  ├─e-
│  │  ├─f
│  │  ├─fallback
│  │  ├─g
│  │  ├─gy
│  │  ├─h
│  │  ├─hy
│  │  ├─i
│  │  ├─i-
│  │  ├─j
│  │  ├─k
│  │  ├─ky
│  │  ├─m
│  │  ├─my
│  │  ├─n
│  │  ├─nn
│  │  ├─ny
│  │  ├─o
│  │  ├─o-
│  │  ├─p
│  │  ├─py
│  │  ├─q
│  │  ├─r
│  │  ├─ry
│  │  ├─s
│  │  ├─sh
│  │  ├─silent
│  │  ├─t
│  │  ├─ts
│  │  ├─ty
│  │  ├─u
│  │  ├─u-
│  │  ├─w
│  │  ├─y
│  │  ├─z
│  │  └─zy
│  └─scripts
├─segmentation-kit (git submodule)
└─target
```

### data

Put input files here.

#### images

- (Required) put fallback image in `fallback` directory
    - this will be the image of ones which should be in other empty directories
- (Optional) put background image in `background` directory
    - useful such as for static body except face
- (Optional) put faces of vowels in `a`, `i`, `u`, `e`, `o` directory
- (Optional) put faces of longer vowels in `a-`, `i-`, `u-`, `e-`, `o-` directory
- (Optional) put faces of N (撥音「ん」) in `nn` directory
- (Optional) put faces of q (促音「っ」) in `q` directory
- (Optional) put faces of silent (無声) in `silent` directory
- (Optional) put faces of consonants in other directory

#### scripts

Put pairs of WAV file (voice) and TXT file (transcript in Japanese) with the same name except file extension.

- Example

```
target
├─source.wav
└─source.txt
```

### target

Output will be produced here.

- `<source>.mp4`: Lip-sync movie without audio
- `<source>-with-audio.mp4`: Lip-sync movie with audio

## Usage

1. Just run `convert_text_to_pronounce.py` as Python code :tada:
