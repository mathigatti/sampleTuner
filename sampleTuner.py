import sys
from aubio import source, pitch
from scipy.io import wavfile
import numpy as np

def noteToFreq(note):
    a = 440 #frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


def samplePitch(filename):
    downsample = 1
    samplerate = 44100 // downsample
    #if len( sys.argv ) > 2: samplerate = int(sys.argv[2])

    win_s = 4096 // downsample # fft size
    hop_s = 512  // downsample # hop size

    s = source(filename, samplerate, hop_s)
    samplerate = s.samplerate

    tolerance = 0.8

    pitch_o = pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    pitches = []
    confidences = []

    # total number of frames read
    total_frames = 0
    while True:
        samples, read = s()
        pitch_i = pitch_o(samples)[0]
        #pitch = int(round(pitch))
        confidence = pitch_o.get_confidence()
        #if confidence < 0.8: pitch = 0.
        #print("%f %f %f" % (total_frames / float(samplerate), pitch, confidence))
        pitches += [pitch_i]
        confidences += [confidence]
        total_frames += read
        if read < hop_s: break

    return noteToFreq(pitches[-1])

def factor(new_f,actual_f):
    distance = actual_f%new_f
    if distance > new_f/2:
        return (new_f*(actual_f//new_f+1))/actual_f        
    else:
        return (new_f*(actual_f//new_f))/actual_f

def speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]


def main():
    filename = sys.argv[1]

    if len(sys.argv) > 2:
        new_freq = sys.argv[2]
    else:
        new_freq = 261.63 # C4 = 261.63 Hz

    actual_freq = samplePitch(filename)

    fs, data = wavfile.read(filename)

    factor_x = factor(new_freq,actual_freq)
    print((new_freq,actual_freq))
    print(factor_x)
    wavfile.write("output.wav",fs,speedx(data, factor_x))

main()