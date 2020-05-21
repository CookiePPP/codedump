from pydub import AudioSegment
import os
directory = "/media/tanarossiter/Samsung PM961/TwiBot/tacotron2/audio"

for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        sound = AudioSegment.from_wav(directory+"/"+filename)
        sound = sound.set_channels(1)
        sound.export(directory+"/"+filename, format="wav")
        continue
    else:
        continue

