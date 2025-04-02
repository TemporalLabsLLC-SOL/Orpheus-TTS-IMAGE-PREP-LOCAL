#!/usr/bin/env python3
import os
import math
import csv
import re
import time
import tkinter as tk
import tkinter.filedialog as fd

from pydub import AudioSegment
import whisper
from datasets import load_dataset, Audio

# --------- PART 1: CREATE INITIAL HF DATASET FROM AUDIO (CHUNKED & TRANSCRIBED) ---------

def split_text_into_sentences(text):
    """
    Split text into sentences using punctuation boundaries.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def process_whisper_segments(segments):
    """
    For each Whisper segment, split the text into sentences and estimate start/end times proportionally.
    Returns a list of tuples: (sentence_text, start_time, end_time)
    """
    sentence_list = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].strip()
        if not seg_text:
            continue
        sentences = split_text_into_sentences(seg_text)
        total_chars = sum(len(s) for s in sentences)
        seg_duration = seg_end - seg_start
        current_time = seg_start
        for s in sentences:
            s_duration = seg_duration * (len(s) / total_chars) if total_chars > 0 else 0
            sentence_list.append((s, current_time, current_time + s_duration))
            current_time += s_duration
    sentence_list.sort(key=lambda x: x[1])
    return sentence_list

def merge_sentences(sentences, min_duration=5.0, max_duration=7.0):
    """
    Merge adjacent sentences so that each chunk is at least min_duration seconds.
    Returns a list of chunks: (merged_text, chunk_start, chunk_end)
    """
    chunks = []
    i = 0
    while i < len(sentences):
        text, start_time, end_time = sentences[i]
        chunk_duration = end_time - start_time
        j = i
        # Merge subsequent sentences if chunk is too short
        while chunk_duration < min_duration and j + 1 < len(sentences):
            j += 1
            text += " " + sentences[j][0]
            end_time = sentences[j][2]
            chunk_duration = end_time - start_time
        chunks.append((text, start_time, end_time))
        i = j + 1
    return chunks

def create_initial_dataset():
    # --- Step 1: Popup file selector ---
    root = tk.Tk()
    root.withdraw()
    file_path = fd.askopenfilename(
        title="Select Long Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        exit(1)

    # --- Step 2: Create folder structure ---
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    main_folder = os.path.join(os.getcwd(), base_name)
    audio_folder = os.path.join(main_folder, "audio")
    transcript_folder = os.path.join(main_folder, "transcripts")
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(transcript_folder, exist_ok=True)

    # --- Step 3: Load the full audio ---
    print("Loading audio file...")
    audio = AudioSegment.from_file(file_path)
    
    # --- Step 4: Run Whisper transcription on the full audio ---
    print("Transcribing with Whisper (this may take a while)...")
    model = whisper.load_model("base")
    result = model.transcribe(file_path, verbose=True)
    segments = result.get("segments", [])
    if not segments:
        print("No segments found in transcription.")
        exit(1)

    # --- Step 5: Process segments to extract sentence-level timings ---
    sentences = process_whisper_segments(segments)
    print(f"Extracted {len(sentences)} sentence candidates.")

    # --- Step 6: Merge sentences to get chunks of ~5-7 seconds ---
    chunks = merge_sentences(sentences, min_duration=5.0, max_duration=7.0)
    print(f"Created {len(chunks)} audio chunk(s) based on sentence boundaries.")

    # --- Step 7: Export chunks and transcripts; create metadata CSV ---
    metadata_csv = os.path.join(main_folder, "metadata.csv")
    with open(metadata_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["audio_file", "transcript"])  # header

        for idx, (chunk_text, chunk_start, chunk_end) in enumerate(chunks):
            start_ms = int(chunk_start * 1000)
            end_ms = int(chunk_end * 1000)
            chunk_audio = audio[start_ms:end_ms]

            chunk_filename = f"chunk_{idx+1:03d}.wav"
            chunk_audio_path = os.path.join(audio_folder, chunk_filename)
            transcript_filename = f"chunk_{idx+1:03d}.txt"
            transcript_path = os.path.join(transcript_folder, transcript_filename)

            chunk_audio.export(chunk_audio_path, format="wav")
            print(f"Exported {chunk_audio_path}")

            with open(transcript_path, "w", encoding="utf-8") as t_file:
                t_file.write(chunk_text)
            print(f"Saved transcript {transcript_path}")

            writer.writerow([chunk_audio_path, chunk_text])
    print(f"Metadata CSV created at {metadata_csv}")

    # --- Step 8: Create HF dataset from CSV and push to Hub ---
    print("Loading CSV as Hugging Face dataset...")
    ds = load_dataset("csv", data_files=metadata_csv)
    ds = ds.cast_column("audio_file", Audio(sampling_rate=16000))
    print("Dataset preview:")
    print(ds)

    repo_name = input("Enter your Hugging Face repository name for the initial dataset (e.g., your-username/your-dataset-name): ").strip()
    print(f"Pushing dataset to {repo_name}...")
    ds.push_to_hub(repo_name)
    print("Initial dataset uploaded successfully.")
    return repo_name  # Return the name so that the tokenization step knows which dataset to load

# --------- PART 2: TOKENIZE THE INITIAL DATASET TO CREATE A SECOND DATASET ---------

def create_tokenized_dataset(my_original_dataset_name, name_to_push_dataset_to):
    import torch
    import torchaudio.transforms as T
    from snac import SNAC
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # Make sure the preferred encoding is UTF-8
    import locale
    locale.getpreferredencoding = lambda: "UTF-8"

    # --- Step 1: Download the original dataset from HF Hub ---
    print("Downloading the original dataset from HF Hub...")
    snapshot_download(
        repo_id=my_original_dataset_name,
        repo_type="dataset",
        revision="main",
        max_workers=64,
    )

    ds = load_dataset(my_original_dataset_name, split="train")
    # In our initial dataset, the audio is stored under "audio_file" and transcript under "transcript"
    ds_sample_rate = ds[0]["audio_file"]["sampling_rate"]

    # --- Step 2: Load SNAC model for audio tokenization ---
    print("Loading SNAC model...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # --- Step 3: Define tokenisation function for audio ---
    def tokenise_audio(waveform):
        waveform = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
        waveform = resample_transform(waveform)
        waveform = waveform.unsqueeze(0).to(device)
        with torch.inference_mode():
            codes = model.encode(waveform)
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + 128266)
            all_codes.append(codes[1][0][2*i].item() + 128266 + 4096)
            all_codes.append(codes[2][0][4*i].item() + 128266 + (2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item() + 128266 + (3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item() + 128266 + (4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item() + 128266 + (5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item() + 128266 + (6*4096))
        return all_codes

    # --- Step 4: Map tokenisation over dataset ---
    def add_codes(example):
        codes_list = None
        try:
            answer_audio = example.get("audio_file")
            if answer_audio and "array" in answer_audio:
                audio_array = answer_audio["array"]
                codes_list = tokenise_audio(audio_array)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
        example["codes_list"] = codes_list
        return example

    print("Tokenizing audio with SNAC...")
    ds = ds.map(add_codes, remove_columns=["audio_file"])

    # --- Step 5: Load text tokenizer and set constants ---
    tokeniser_length = 128256
    start_of_text = 128000
    end_of_text = 128009

    start_of_speech = tokeniser_length + 1
    end_of_speech = tokeniser_length + 2

    start_of_human = tokeniser_length + 3
    end_of_human = tokeniser_length + 4
    start_of_ai = tokeniser_length + 5
    end_of_ai =  tokeniser_length + 6
    pad_token = tokeniser_length + 7
    audio_tokens_start = tokeniser_length + 10

    tokenizer_name = "canopylabs/orpheus-3b-0.1-pretrained"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    import os
    num_proc = os.cpu_count() - 2

    # --- Step 6: Remove samples with no audio codes ---
    ds = ds.filter(lambda x: x["codes_list"] is not None)
    ds = ds.filter(lambda x: len(x["codes_list"]) > 0)

    # --- Step 7: Remove duplicate frames in the codes_list ---
    def remove_duplicate_frames(example):
        vals = example["codes_list"]
        if len(vals) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")
        result = vals[:7]
        for i in range(7, len(vals), 7):
            current_first = vals[i]
            previous_first = result[-7]
            if current_first != previous_first:
                result.extend(vals[i:i+7])
        example["codes_list"] = result
        return example

    ds = ds.map(remove_duplicate_frames, num_proc=num_proc)

    # --- Step 8: Create input_ids by combining text and audio tokens ---
    def create_input_ids(example):
        # Use the transcript text from the original dataset
        text_ids = tokenizer.encode(example["transcript"], add_special_tokens=True)
        text_ids.append(end_of_text)
        example["text_tokens"] = text_ids
        input_ids = (
            [start_of_human]
            + example["text_tokens"]
            + [end_of_human]
            + [start_of_ai]
            + [start_of_speech]
            + example["codes_list"]
            + [end_of_speech]
            + [end_of_ai]
        )
        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        return example

    print("Creating input IDs for tokenized dataset...")
    ds = ds.map(create_input_ids, num_proc=num_proc, remove_columns=["transcript", "codes_list"])

    # --- Step 9: Keep only necessary columns and push to Hub ---
    columns_to_keep = ["input_ids", "labels", "attention_mask"]
    columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
    ds = ds.remove_columns(columns_to_remove)

    print("Pushing tokenized dataset to HF Hub...")
    ds.push_to_hub(name_to_push_dataset_to)
    print("Tokenized dataset uploaded successfully.")

def main():
    print("===== PART 1: Creating initial dataset =====")
    initial_repo = create_initial_dataset()
    print("\nPlease wait while the initial dataset is settled on HF Hub.")
    print("When ready, press Enter to proceed with tokenization.")
    input()

    print("\n===== PART 2: Creating tokenized dataset =====")
    # Ask the user to provide the original dataset name and the target name for tokenized dataset.
    my_original_dataset_name = input(f"Enter the original dataset name (should be {initial_repo} or similar): ").strip()
    name_to_push_dataset_to = input("Enter the tokenized dataset repository name (e.g., <my-namespace>/your-dataset-tokenised): ").strip()
    create_tokenized_dataset(my_original_dataset_name, name_to_push_dataset_to)

if __name__ == "__main__":
    main()
