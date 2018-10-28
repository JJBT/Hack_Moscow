import pydub
import speech_recognition as sr
from monkeylearn import MonkeyLearn
import os
from Constants import *


def get_wav_speech(mp3_file_name, wav_file_name):
    sound = pydub.AudioSegment.from_mp3(mp3_file_name)
    sound.export(wav_file_name, format="wav")


def speech_to_text(wav_file_name):
    recognizer = sr.Recognizer()
    file = sr.AudioFile(wav_file_name)
    with file as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)


def monkey_text_analyzer(speech_to_text_data):
    ml = MonkeyLearn('3d6dc8322110618ae2793d7c4e732f36127bc9e5')
    model_id = 'cl_pi3C7JiL'
    result = ml.classifiers.classify(model_id, [speech_to_text_data])
    return result.body[0]['classifications'][0]['tag_name'], result.body[0]['classifications'][0]['confidence']


def get_swear_words(swear_words_file):
    swear_words = []
    swear_words_file = open(swear_words_file, 'r')
    for word in swear_words_file:
        swear_words.append(word.strip())
    swear_words_file.close()
    return swear_words


def detect_swear_words(swear_words, speech_to_text_data):
    rec_words = speech_to_text_data.split()
    swear_rec_words = [word for word in rec_words if ((word in swear_words) or (word.find('*') != -1))]
    if swear_rec_words:
        return True
    else:
        return False


def check_text(wav_file_name, swear_words_file, real_text='', wav_file_name2='None'):
    if wav_file_name2 == 'None':
        speech_to_text_data = speech_to_text(wav_file_name=wav_file_name)
        emotion_label, confidence = monkey_text_analyzer(speech_to_text_data=speech_to_text_data)

        swear_words = get_swear_words(swear_words_file=swear_words_file)
        is_swear = detect_swear_words(swear_words=swear_words, speech_to_text_data=speech_to_text_data)
        print("The original text of the audio: ", real_text)
        print("Recognized text: ", speech_to_text_data)
        print('Label: ', emotion_label, '\n', 'Confidence: ', confidence, '\n')
        if is_swear:
            print('Rude words were used')
        else:
            print('There were no rude words')
    else:

        speech_to_text_data1 = speech_to_text(wav_file_name=wav_file_name)
        speech_to_text_data2 = speech_to_text(wav_file_name=wav_file_name2)
        speech_to_text_data = speech_to_text_data1 + ' ' + speech_to_text_data2
        emotion_label, confidence = monkey_text_analyzer(speech_to_text_data=speech_to_text_data)
        swear_words = get_swear_words(swear_words_file=swear_words_file)
        is_swear = detect_swear_words(swear_words=swear_words, speech_to_text_data=speech_to_text_data)
        print("The original text of the audio: ", real_text)
        print("Recognized text: ", speech_to_text_data)
        print('Label: ', emotion_label, '\n', 'Confidence: ', confidence, '\n')
        if is_swear:
            print('Rude words were used')
        else:
            print('There were no rude words')
            print('\n\n')

    return 0


def start_check(path):
    os.chdir("../../")
    os.chdir(path)
    for file in os.listdir('.'):
        check_text(file, '../../swear_words.txt')
