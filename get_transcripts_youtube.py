from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from concurrent.futures import ThreadPoolExecutor
from retrying import retry

# Function to find and fetch video transcript from YouTube.

#@retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def find_transcript(video_id):
    try:
        # Get the list of all transcripts available for the video.
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Check if human-generated English transcript is available.
        for transcript in transcript_list:
            if transcript.language_code == 'en' and not transcript.is_generated:
                # Return fetched transcript text after joining all parts.
                return ' '.join([part['text'] for part in transcript.fetch()])

        # Check if auto-generated English transcript is available.
        for transcript in transcript_list:
            if transcript.language_code == 'en' and transcript.is_generated:
                # Return fetched transcript text after joining all parts.
                return ' '.join([part['text'] for part in transcript.fetch()])

        # Check if human-generated non-English transcript is available.
        for transcript in transcript_list:
            if not transcript.is_generated:
                try:
                    # Try to translate transcript to English.
                    trans_text = ' '.join([part['text'] for part in transcript.translate('en').fetch()])
                    return trans_text
                except IndexError:
                    # If translation fails, return the transcript in original language.
                    return ' '.join([part['text'] for part in transcript.fetch()])

        # If no preferred transcripts available, return the first available transcript.
        for transcript in transcript_list:
            try:
                # Try to translate transcript to English.
                trans_text = ' '.join([part['text'] for part in transcript.translate('en').fetch()])
                if trans_text == '':
                    # If translation is empty, return the transcript in original language.
                    return ' '.join([part['text'] for part in transcript.fetch()])
                else:
                    return trans_text
            except IndexError:
                # If translation fails, return the transcript in original language.
                return ' '.join([part['text'] for part in transcript.fetch()])

    except TranscriptsDisabled:
        # Return a message if transcripts are disabled for the video.
        return "No transcription"
    except Exception as e:
        # Log any other exceptions encountered during transcript fetching.
        print("Video_id: " + video_id, f"An error occurred while retrieving the transcription from YouTube: {str(e)}")
        return "Error"


# Function to use multithreading for fetching video transcripts.
def retrieve_captions(video_ids, func):
    # Use ThreadPoolExecutor as a context manager to handle thread pool.
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Process the function passed in 'func' for each video ID.
        return list(tqdm(executor.map(func, video_ids), total=len(video_ids), desc='Retrieving the transcriptions from YouTube'))
