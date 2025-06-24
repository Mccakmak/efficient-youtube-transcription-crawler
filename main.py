# FileNotFoundError: [WinError 2] The system cannot find the file specified
# https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/

# TypeError: argument of type 'NoneType' is not iterable when importing whisper -->
# pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

# Python 3.9.10
# pip install googletrans==4.0.0rc1

# Streaming API error
# Run pip uninstall pytube
# Run pip uninstall pytube3
# Run python -m pip install git+https://github.com/pytube/pytube


#import get_transcripts_youtube as gty
import find_no_caption_whisper as fcw
import translate as tr
import merge as mg
import pandas as pd
from pandas.errors import ParserError


def main():
    # Input file name
    file_name = "root_taiwan_election_unique"

    # Read file
    try:
        df = pd.read_csv('input/' + file_name + ".csv")
    except ParserError:
        try:
            df = pd.read_excel('input/' + file_name + ".xlsx")
        except FileNotFoundError:
            print("Neither a .csv nor .xlsx file found with the name:", file_name)
    except FileNotFoundError:
        print("File not found:", file_name + ".csv")

    # STEP 1
    # Retrieve already available captions

    df = df[df['video_id'].notna()]
    res = gty.retrieve_captions(df.video_id.to_list(), gty.find_transcript)

    df['transcription'] = res
    df.to_csv('temp output/'+file_name+"_transcription.csv", index=False)

    # STEP 2
    #Speech recognition using Whisper
    fcw.speech_recognition(file_name+"_transcription")

    # Translation
    tr.translate_en(file_name+ "_transcription_whisper", 'whisper')
    #tr.translate_en(file_name + "_transcription",'yt_api')

    # Merge
    mg.main(file_name)


if __name__ == "__main__":
    main()
