#!/usr/bin/env python3
import os
import math
import csv
import re
import tkinter as tk
import tkinter.filedialog as fd

from pydub import AudioSegment
import whisper
from datasets import load_dataset, Audio

def split_text_into_sentences(text):
    """
    Splits text into sentences using punctuation as boundaries.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def process_whisper_segments(segments):
    """
    For each Whisper segment, split the text into sentences and estimate start/end times
    proportionally based on character counts.
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
            # Estimate duration proportional to sentence length (in seconds)
            if total_chars > 0:
                s_duration = seg_duration * (len(s) / total_chars)
            else:
                s_duration = 0
            sentence_list.append((s, current_time, current_time + s_duration))
            current_time += s_duration
    # Ensure sentences are sorted by start time (should already be)
    sentence_list.sort(key=lambda x: x[1])
    return sentence_list

def merge_sentences(sentences, min_duration=5.0, max_duration=7.0):
    """
    Merge adjacent sentences so that each chunk is at least min_duration seconds.
    The function will never cut a sentence in half.
    If a sentence alone exceeds max_duration, it is kept as is.
    Returns a list of chunks: (merged_text, chunk_start, chunk_end)
    """
    chunks = []
    i = 0
    while i < len(sentences):
        text, start_time, end_time = sentences[i]
        chunk_duration = end_time - start_time
        j = i
        # Merge subsequent sentences if the chunk is too short
        while chunk_duration < min_duration and j + 1 < len(sentences):
            j += 1
            text += " " + sentences[j][0]
            end_time = sentences[j][2]
            chunk_duration = end_time - start_time
        # We could try to avoid overshooting max_duration if more than one sentence was merged,
        # but we must not cut a sentence. So if the merged chunk is longer than max_duration,
        # we leave it as is.
        chunks.append((text, start_time, end_time))
        i = j + 1
    return chunks

def main():
    # --- Step 1: Popup file selector ---
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = fd.askopenfilename(
        title="Select Long Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    # --- Step 2: Create main folder and subfolders ---
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
    print("Running Whisper transcription (this may take a while)...")
    model = whisper.load_model("base")
    result = model.transcribe(file_path, verbose=True)
    segments = result.get("segments", [])
    if not segments:
        print("No segments found in transcription.")
        return

    # --- Step 5: Process segments to extract sentence-level timings ---
    sentences = process_whisper_segments(segments)
    print(f"Extracted {len(sentences)} sentence candidates from transcription.")

    # --- Step 6: Merge sentences to achieve chunks of ~5-7 seconds ---
    chunks = merge_sentences(sentences, min_duration=5.0, max_duration=7.0)
    print(f"Created {len(chunks)} audio chunk(s) based on sentence boundaries.")

    # --- Step 7: Process each chunk: export audio, save transcript, and record metadata ---
    metadata_csv = os.path.join(main_folder, "metadata.csv")
    with open(metadata_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["audio_file", "transcript"])  # header

        for idx, (chunk_text, chunk_start, chunk_end) in enumerate(chunks):
            # Ensure times are in milliseconds for pydub
            start_ms = int(chunk_start * 1000)
            end_ms = int(chunk_end * 1000)
            chunk_audio = audio[start_ms:end_ms]

            chunk_filename = f"chunk_{idx+1:03d}.wav"
            chunk_audio_path = os.path.join(audio_folder, chunk_filename)
            transcript_filename = f"chunk_{idx+1:03d}.txt"
            transcript_path = os.path.join(transcript_folder, transcript_filename)

            # Export audio chunk
            chunk_audio.export(chunk_audio_path, format="wav")
            print(f"Exported audio chunk: {chunk_audio_path}")

            # Save transcript text
            with open(transcript_path, "w", encoding="utf-8") as t_file:
                t_file.write(chunk_text)
            print(f"Saved transcript: {transcript_path}")

            # Write row to CSV (use absolute or relative path as needed)
            writer.writerow([chunk_audio_path, chunk_text])

    print(f"Metadata CSV created at {metadata_csv}")

    # --- Step 8: Create a Hugging Face Dataset from the CSV ---
    print("Creating Hugging Face dataset from CSV...")
    dataset = load_dataset("csv", data_files=metadata_csv)
    # Cast the audio_file column to Audio feature (adjust sampling_rate if needed)
    dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000))
    print("Dataset preview:")
    print(dataset)

    # --- Step 9: Push the dataset to the Hugging Face Hub ---
    repo_name = input("Enter your Hugging Face repository name (e.g., your-username/your-dataset-name): ").strip()
    print(f"Pushing dataset to {repo_name}...")
    dataset.push_to_hub(repo_name)
    print("Dataset uploaded successfully.")

if __name__ == "__main__":
    main()
