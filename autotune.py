from functools import partial
from pathlib import Path
import argparse
import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf





def pitch_tracking(audio, sr, plot=False):

    #set these values since you need them for the spectogram
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
      frame_length=frame_length, hop_length=hop_length, sr=sr,
      fmin=fmin, fmax=fmax)

    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    time_points = librosa.times_like(stft, sr=sr, hop_length=hop_length)
    log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(time_points, f0, label='original pitch', color='cyan', linewidth=2)
    ax.legend(loc='upper right')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [M:SS]')
    plt.savefig('audio.png', dpi=300, bbox_inches='tight')



def main ():
    #parse the command line for the audio file
    parser = argparse.ArgumentParser()

    #adds the file name as a required argument
    parser.add_argument("filename_arg",
                    help='the name of the audiofile you will open')
    
    #parse arguments, will automatically error if right args not passed
    # you are passing the name of the file)
    args = parser.parse_args()

    #get the filepath of .wav file
    filepath = Path(args.first_argument)

    #load audio file
    y, sr = librosa.load(str(filepath), sr=None, mono=False)

    #perform the tracking/graphing
    pitch_tracking(y, sr, args.plot)


    


        
