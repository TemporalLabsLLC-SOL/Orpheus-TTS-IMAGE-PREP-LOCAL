# Orpheus-TTS-IMAGE-PREP-LOCAL
Quickly prepare and upload your image dataset to huggingface all with one simple script to prepare data to be translated for eventual fine-tuning of Orpheus-TTS

## SETUP

Sign-in on HuggingFace
`https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained`
  - Sign Gated Permission Agreement
#Create Full HF Token - Prepare to Paste in Terminal

# SETUP ENVIRONMENT AND NAVIGATE TO INTENDED DIRECTORY ROOT

# Clone main Orepheus-TTS repo
`git clone https://github.com/canopyai/Orpheus-TTS.git`

# Navigate and install packages
`cd Orpheus-TTS && pip install orpheus-speech`

vllm pushed a slightly buggy version on March 18th so some bugs are being resolved by reverting to `pip install vllm==0.7.3` after pip install orpheus-speech

