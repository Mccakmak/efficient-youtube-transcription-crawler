# High Speed YouTube Transcription Collection

This repository provides a robust system for collecting, generating, and translating transcripts from YouTube videos using a combination of parallel processing, Whisper speech-to-text, and multilingual translation models.

The methodology significantly accelerates transcript acquisition from large video datasets, addressing common issues such as unavailable captions, language barriers, and API rate limits. It supports tasks ranging from crawling, transcription, translation, and merging—all executed efficiently via multiprocessing and GPU acceleration.

---

## 🔍 Features

- 📺 Collects available captions from YouTube using [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
- 🧠 Uses OpenAI's Whisper (and `faster-whisper`) to transcribe videos without captions
- 🌍 Translates non-English transcripts to English via `googletrans` or M2M100
- ⚡ Fully parallelized using Python's `multiprocessing` and `concurrent.futures` for both I/O and CPU-bound tasks
- 📦 Modular scripts for collection, transcription, translation, and merging
- 📈 Scales with GPUs for Whisper transcription and translation efficiency

---

## 📂 Scripts

| Script | Description |
|--------|-------------|
| `get_transcripts_youtube.py` | Uses YouTube Transcript API to get available captions |
| `find_no_caption_whisper.py` | Applies Whisper to videos missing captions |
| `audio_to_transcript_only.py` | Isolated audio transcription using Whisper |
| `translate.py` | Translates non-English transcripts |
| `merge.py` | Merges and replaces missing or translated transcripts |
| `main.py` | Example pipeline to execute all steps |

---

## 📚 Citation

If you use this project in your research, please cite the following works:

```bibtex
@inproceedings{cakmak2024high,
  title     = {High-speed transcript collection on multimedia platforms: Advancing social media research through parallel processing},
  author    = {Cakmak, Mert Can and Agarwal, Nitin},
  booktitle = {2024 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
  pages     = {857--860},
  year      = {2024},
  organization = {IEEE}
}

@inproceedings{cakmak2023adopting,
  title     = {Adopting parallel processing for rapid generation of transcripts in multimedia-rich online information environment},
  author    = {Cakmak, Mert Can and Okeke, Obianuju and Spann, Billy and Agarwal, Nitin},
  booktitle = {2023 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
  pages     = {832--837},
  year      = {2023},
  organization = {IEEE}
}