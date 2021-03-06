import pyaudio
import wave
import time
import sys


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = time.strftime("%Y%m%d_%H%M%S")+"_label.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("")
print("* *********")
print("* RECORDING")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    print ".",
    sys.stdout.flush()
    data = stream.read(CHUNK)
    frames.append(data)

print("")
print("* END")
print("* *********")
print("")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
