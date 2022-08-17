import imp
from gtts import gTTS
from playsound import playsound
import os
import time


def speak_name(name):

  # sex = json_list[i]['sex']
  text = "Xin chào {name}".format(name=name)
  return text

def speak(text):
    tts = gTTS(text, lang='vi', slow=False)
    tts.save('speak.mp3')
    playsound('speak.mp3')
    time.sleep(2)
    os.remove('speak.mp3')