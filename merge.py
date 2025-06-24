import pandas as pd


def main(filename):
    df_whisper = pd.read_csv('temp output/' + filename + "_transcription_whisper_translated.csv")
    df_all = pd.read_csv('temp output/' + filename + "_transcription.csv")

    # change old transcription which is 'no transcription' to new transcription
    for video_id_whisper in df_whisper.video_id:
        index_val = df_whisper[df_whisper['video_id'] == video_id_whisper]['transcription'].index.values[0]
        new_transcription = df_whisper[df_whisper['video_id'] == video_id_whisper]['transcription'][index_val]
        df_all.loc[(df_all['video_id'] == video_id_whisper), 'transcription'] = new_transcription

    # Removing the rest transcription that has no transcriptions.
    df_all = df_all[df_all.transcription != "No transcription"]
    df_all = df_all[df_all.transcription != "Not Video"]
    df_all = df_all[df_all.transcription != "Error"]

    # Save
    df_all.to_csv('final output/' + filename + "_final_transcript.csv", index=False)


if __name__ == '__main__':
    main("scs_recommended_D50_T60_common_w0_video")
