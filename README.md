# Orpheus-TTS-IMAGE-PREP-LOCAL
Quickly prepare and upload your image dataset to huggingface all with one simple script to prepare data to be translated for eventual fine-tuning of Orpheus-TTS

## SETUP

Sign-in on HuggingFace
`https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained`
  - Sign Gated Permission Agreement
`huggingface-cli login <YOUR_HF_TOKEN>`

# SETUP ENVIRONMENT AND NAVIGATE TO INTENDED DIRECTORY ROOT

## Clone main Orepheus-TTS repo
`git clone https://github.com/canopyai/Orpheus-TTS.git`

## Navigate and install packages
`cd Orpheus-TTS && pip install orpheus-speech`

vllm pushed a slightly buggy version on March 18th so some bugs are being resolved by reverting to `pip install vllm==0.7.3` after pip install orpheus-speech

# INSTALL THIS REPO AND SCRIPT

`git clone https://github.com/TemporalLabsLLC-SOL/Orpheus-TTS-IMAGE-PREP-LOCAL.git`

### Place process_audio.py within `./Orpheus-TTS/pretrain/`

`cd pretrain`

## Start Audio Processing

`python3 audio_process.py`

### A Popup will request you to select an audio file - Select Any Audio File
### Simply Give the names of your databases in the expected format of <YOUR_HF_USERNAME>/<ANY_NAME_YOU_WANT>
#### This Creates The DataBases


