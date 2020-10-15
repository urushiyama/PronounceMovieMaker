# Pronounce Movie Maker

## Requirements

- some environment to run python script (tested by [Anaconda](https://www.anaconda.com/products/individual))
- some third-party python libraries (run `pip install -r requirements.txt` to use)
- [MeCab](https://taku910.github.io/mecab/#download)
- some environment to run perl script (tested by [ActivePerl](https://www.activestate.com/products/perl/downloads/))

## Directory Structure

```
PronounceMovieMaker
├─data
│  ├─images
│  │  ├─consonants
│  │  │  ├─b
│  │  │  ├─by
│  │  │  ├─ch
│  │  │  ├─d
│  │  │  ├─dy
│  │  │  ├─f
│  │  │  ├─g
│  │  │  ├─gy
│  │  │  ├─h
│  │  │  ├─hy
│  │  │  ├─k
│  │  │  ├─ky
│  │  │  ├─m
│  │  │  ├─my
│  │  │  ├─n
│  │  │  ├─ny
│  │  │  ├─p
│  │  │  ├─py
│  │  │  ├─r
│  │  │  ├─ry
│  │  │  ├─s
│  │  │  ├─sh
│  │  │  ├─t
│  │  │  ├─ts
│  │  │  ├─ty
│  │  │  ├─y
│  │  │  ├─z
│  │  │  └─zy
│  │  ├─fallback
│  │  ├─mora
│  │  │  ├─N
│  │  │  └─q
│  │  ├─semivowels
│  │  │  ├─j
│  │  │  └─w
│  │  ├─silent
│  │  └─vowels
│  │      ├─a-
│  │      ├─e
│  │      ├─e-
│  │      ├─i
│  │      ├─i-
│  │      ├─o
│  │      ├─o-
│  │      ├─u
│  │      └─u-
│  └─scripts
├─segmentation-kit (submodule)
└─target
```
