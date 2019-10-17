import sys
from aubio import source, pitch
from scipy.io import wavfile
import numpy as np
import librosa

notes2frequencies = {"G#9/Ab9":13289.75, "G9":12543.85, "F#9/Gb9":11839.82, "F9":11175.30, "E9":10548.08, "D#9/Eb9":9956.06, "D9":9397.27, "C#9/Db9":8869.84, "C9":8372.02, "B8":7902.13, "A#8/Bb8":7458.62, "A8":7040.00, "G#8/Ab8":6644.88, "G8":6271.93, "F#8/Gb8":5919.91, "F8":5587.65, "E8":5274.04, "D#8/Eb8":4978.03, "D8":4698.64, "C#8/Db8":4434.92, "C8":4186.01, "B7":3951.07, "A#7/Bb7":3729.31, "A7":3520.00, "G#7/Ab7":3322.44, "G7":3135.96, "F#7/Gb7":2959.96, "F7":2793.83, "E7":2637.02, "D#7/Eb7":2489.02, "D7":2349.32, "C#7/Db7":2217.46, "C7":2093.00, "B6":1975.53, "A#6/Bb6":1864.66, "A6":1760.00, "G#6/Ab6":1661.22, "G6":1567.98, "F#6/Gb6":1479.98, "F6":1396.91, "E6":1318.51, "D#6/Eb6":1244.51, "D6":1174.66, "C#6/Db6":1108.73, "C6":1046.50, "B5":987.77, "A#5/Bb5":932.33, "A5":880.00, "G#5/Ab5":830.61, "G5":783.99, "F#5/Gb5":739.99, "F5":698.46, "E5":659.26, "D#5/Eb5":622.25, "D5":587.33, "C#5/Db5":554.37, "C5":523.25, "B4":493.88, "A#4/Bb4":466.16, "A4 concert pitch":440.00, "G#4/Ab4":415.30, "G4":392.00, "F#4/Gb4":369.99, "F4":349.23, "E4":329.63, "D#4/Eb4":311.13, "D4":293.66, "C#4/Db4":277.18, "C4 (middle C)":261.63, "B3":246.94, "A#3/Bb3":233.08, "A3":220.00, "G#3/Ab3":207.65, "G3":196.00, "F#3/Gb3":185.00, "F3":174.61, "E3":164.81, "D#3/Eb3":155.56, "D3":146.83, "C#3/Db3":138.59, "C3":130.81, "B2":123.47, "A#2/Bb2":116.54, "A2":110.00, "G#2/Ab2":103.83, "G2":98.00, "F#2/Gb2":92.50, "F2":87.31, "E2":82.41, "D#2/Eb2":77.78, "D2":73.42, "C#2/Db2":69.30, "C2":65.41, "B1":61.74, "A#1/Bb1":58.27, "A1":55.00, "G#1/Ab1":51.91, "G1":49.00, "F#1/Gb1":46.25, "F1":43.65, "E1":41.20, "D#1/Eb1":38.89, "D1":36.71, "C#1/Db1":34.65, "C1":32.70, "B0":30.87, "A#0/Bb0":29.14, "A0":27.50}
frequencies2notes = {v:k for k,v in notes2frequencies.items()}
cDistance = {"F#":-6, "F":-5, "E":-4, "D#":-3, "D":-2, "C#":-1, "C":0, "B":1, "A#":2, "A":3,"G#":4,"G":5}

def noteToFreq(note):
    a = 440 #frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))

def samplePitch(filename):
    downsample = 1
    samplerate = wavfile.read(filename)[0]
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
    pitches = list(filter(lambda x : x != 0, pitches))
    pitches = sorted(pitches)
    outliers = 0.1 # Remove extreme values
    pitches = pitches[int(len(pitches)*outliers):-int(len(pitches)*outliers)]
    #print(pitches)
    return noteToFreq(np.array(pitches).mean())

def factor(new_f,actual_f):
    return new_f/actual_f

def speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]

def closestNote(freq):
    freqs = sorted(list(frequencies2notes.keys()))
    #print(freqs)
    closest = freqs[0]
    for freq_new in freqs[1:]:
        if freq < freq_new:
            if closest - freq < freq_new - freq:
                return frequencies2notes[closest]
            else:
                return frequencies2notes[freq_new]
        closest = freq_new
    raise Exception()

def extractNote(note):
    if note[1].isnumeric():
        return note[0]
    else:
        return note[0:2]

def c_tuning(audio_file, note):
    n_steps = cDistance[extractNote(note)]

    y, sr = librosa.load(audio_file, sr=16000) # y is a numpy array of the wav file, sr = sample rate
    y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    librosa.output.write_wav('output.wav', y_shifted, sr)

def main():
    filename = sys.argv[1]

    actual_freq = samplePitch(filename)

    print("ACTUAL FREQ", actual_freq)

    closest_note = closestNote(actual_freq)

    print("CLOSEST NOTE", closest_note)

    fs, data = wavfile.read(filename)

    factor_x = factor(notes2frequencies[closest_note],actual_freq)

    print("FACTOR X", factor_x)

    wavfile.write("output.wav",fs,speedx(data, factor_x))

    c_tuning("output.wav", closest_note)    


main()