import os
import csv
from tqdm import tqdm
from faster_whisper import WhisperModel
import translate as tr

def transcribe_files_in_folder(folder_path, output_csv):
    model = WhisperModel("large-v2", device="cuda", compute_type="float16", device_index=[0, 1, 2, 3, 4, 5, 6, 7])

    # Prepare the CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["video_id", "transcription"])

        # List all files in the folder
        files = [f for f in os.listdir(folder_path) if
                 f.endswith(".mp3")]  # Modify this if you have other audio formats

        # Iterate over all files with a progress bar
        for filename in tqdm(files, desc="Transcribing files"):
            file_path = os.path.join(folder_path, filename)

            # Transcribe the audio file
            segments, info = model.transcribe(file_path)
            transcription = ""
            for segment in segments:
                transcription += segment.text

            # Write to CSV
            writer.writerow([filename, transcription])

if __name__ == "__main__":
    folder_path = "audios"
    output_csv = "transcriptions.csv"
    #transcribe_files_in_folder(folder_path, output_csv)
    tr.translate_en("taiwan_transcriptions")