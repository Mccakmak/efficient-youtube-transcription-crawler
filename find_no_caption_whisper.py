import concurrent.futures
import pandas as pd
from tqdm import tqdm
import os
import random
import time
import subprocess
from faster_whisper import WhisperModel

# Ensure proper spawn start method (especially on Windows)
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

# Download audio from YouTube using yt-dlp.
def get_audio(url):
    try:
        audio_folder_path = "audios"
        os.makedirs(audio_folder_path, exist_ok=True)
        video_id = url.split("v=")[-1]
        output_audio = os.path.join(audio_folder_path, f"{video_id}.mp3")
        if os.path.exists(output_audio):
            print(f"Audio already exists for {video_id}, skipping download.")
            return output_audio
        cookies_path = "cookies.txt"
        if not os.path.exists(cookies_path):
            print(f"Cookies file not found at {cookies_path}. Skipping {url}.")
            return 'no_audio_path'
        time.sleep(random.uniform(1.0, 2.0))
        command = [
            "yt-dlp",
            "--cookies", cookies_path,
            "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", output_audio,
            url
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"yt-dlp failed for {url}:\n{result.stderr}")
            return 'no_audio_path'
        return output_audio
    except Exception as exc:
        print(f"Error downloading audio for {url}: {type(exc).__name__}: {exc}")
        return 'no_audio_path'

# Download audios concurrently.
def get_audio_wrapper(video_ids):
    audio_files = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        future_to_vid = {executor.submit(get_audio, f"https://www.youtube.com/watch?v={video_id}"): video_id
                         for video_id in video_ids}
        for future in tqdm(concurrent.futures.as_completed(future_to_vid),
                           total=len(video_ids), desc="Downloading audios"):
            video_id = future_to_vid[future]
            try:
                audio_files[video_id] = future.result()
            except Exception as exc:
                print(f"Error processing video {video_id}: {exc}")
    return audio_files

# Helper to call transcribe with error handling.
def transcribe_single(model, file_path, beam_size):
    try:
        segments, _ = model.transcribe(file_path, beam_size=beam_size, vad_filter=True)
        return [{"text": segment.text.strip()} for segment in segments]
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
        return 'no audio'

# Worker function for a given GPU.
# Each process loads its own model (pinned to one GPU) and processes its audio files sequentially.
def worker_transcribe(audio_subset, device_index, beam_size=2):
    # Create a model instance pinned to a single GPU.
    model = WhisperModel(
        model_size_or_path="medium",
        device="cuda",
        device_index=[device_index],
        compute_type="int8_float16",
        num_workers=8
    )
    results = {}
    # Process audio files sequentially.
    for video_id, file_path in audio_subset.items():
        if file_path == 'no_audio_path':
            results[video_id] = 'no audio'
        else:
            res = transcribe_single(model, file_path, beam_size)
            results[video_id] = res
    return results

# Run transcription across multiple GPUs.
def run_model_multi_gpu(audio_files, num_gpus=8, beam_size=3):
    # Split the audio_files dict into roughly equal parts for each GPU.
    audio_splits = [{} for _ in range(num_gpus)]
    for i, (video_id, file_path) in enumerate(audio_files.items()):
        audio_splits[i % num_gpus][video_id] = file_path

    all_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(worker_transcribe, audio_splits[i], i, beam_size)
                   for i in range(num_gpus)]
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=num_gpus, desc="Transcribing on GPUs"):
            try:
                all_results.update(future.result())
            except Exception as e:
                print(f"Worker failed: {e}")
    return all_results

# Main pipeline: download audios, run transcriptions, and save to CSV.
def speech_recognition(filename):
    df = pd.read_csv(os.path.join('temp output', filename + ".csv"))
    no_cap_video_ids = df[df['transcription'] == 'No transcription']['video_id'].unique()
    print(f"{len(no_cap_video_ids)} videos do not have transcription available. Generating transcriptions...")
    audio_files = get_audio_wrapper(no_cap_video_ids)
    results = run_model_multi_gpu(audio_files, num_gpus=8, beam_size=3)
    transcriptions = {}
    for video_id, segments in results.items():
        if segments == 'no audio':
            transcriptions[video_id] = 'Not Video'
        else:
            text = ''.join([seg["text"] for seg in segments])
            transcriptions[video_id] = text
    pd.DataFrame({
        'video_id': list(transcriptions.keys()),
        'transcription': list(transcriptions.values())
    }).to_csv(os.path.join('temp output', f'{filename}_whisper.csv'), index=False)

if __name__ == "__main__":
    file_name = "scs_recommended_D50_T3_common_w0_video"
    speech_recognition(file_name + "_transcription")
