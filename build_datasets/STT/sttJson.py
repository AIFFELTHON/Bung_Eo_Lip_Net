import os 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './key.json'

#json 파일 형식 맞추기

def transcribe_gcs_with_word_time_offsets(gcs_uri):
    """Transcribe the given audio file asynchronously and output the word time
    offsets."""
    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ko-KR",
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    result = operation.result(timeout=90)
    jsonTry = []
    jsonList = []

    for result in result.results:
        alternative = result.alternatives[0]
        jsonObject = {"transcript":alternative.transcript,"confidence":alternative.confidence,"words":[]}
        jsonTry.append(jsonObject)
        jsonList = []
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time

            #print(f"Word: {word}, start_time: {start_time.total_seconds()}, end_time: {end_time.total_seconds()}")
            jsonWordType = {"end_time": (end_time.total_seconds()), "start_time": (start_time.total_seconds()), "word": (word)} 
            jsonList.append(jsonWordType)
 
        jsonObject["words"] = jsonList 
    
    print(jsonTry)
    return jsonTry
