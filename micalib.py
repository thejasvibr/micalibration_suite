# -*- coding: utf-8 -*-
""" Module with the required functions to do calibrations 
of the various mics used in the 2018 field season. 

The playback sequence consisted of 5 upward linear sweeps and 
82 single tone playbacks from 10-92 kHz. Each playback happened in a 
0.7 second section. 

The calibrations were done on 2 days (2018-07-16 and 2019-01-10).
There are a bunch of common mics across the two days. The naming format for 
the recordings is:

2018-07-16:
    SANKEN_9_1538_gaindB_24_azimuth_angle_0_2018-07-17_19-03-35.wav
    MIC_SERIAL#_gaindB_NUMBER_azimuth_angle_NUMBER_YYYY-MM-DD_hh-mm-ss.wav

    HOwever, it's important to note that any gain values that are not 
    6,30 or 60 dB need to be replaced with the *correct* values because
    I'd assumed these values. Later measurements showed the actual gain
    values to be something completely different. 
    
2019-01-10:
    There are two formats 
    SANKEN_1539_gaindB_30_azimuth_angle_0_2019-01-11_11-47-24.wav    
    
    MIC_SERIAL#_gaindB_NUMBER_azimuth_angle_NUMBER_YYYY-MM-DD_hh-mm-ss.wav
                
    AND...
    SMP4_old_gaindB_3marksfrom30dB_azimuth_angle_0_2019-01-11_16-57-30.wav
    
    MIC_SERIAL#_gaindB_DESCRIPTION_azimuth_angle_NUMBER_YYYY-MM-DD_hh-mm-ss.wav

    Here we see the difference in the gaindB naming. The '3marksfrom30dB' refer
    to a clockwise counting of the marks starting with 1 and excluding the 
    30 dB. 
    

REFERENE MIC:
A GRAS 1/4th'' was used to perform the reference microphone calibrations. 
This mic was available only on the 2019-01-10 session because during the
field season the GRAS mic setup that I took did not work for whatever reason. 

This means I will need to use one the SANKENs as my reference mic for the 
2018-07-16 recording session - I hope this doesn't introduce too much difference. 

                
    
    


Created on Wed Mar 20 17:04:14 2019

@author: tbeleyur
"""
import warnings

import soundfile as sf
import numpy as np 
import scipy.signal as signal 
import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 10000



def get_pure_tone_snippets(mic_rec, fs=192000):
    ''' Extracts the pure tones given 100ms pure tone playbacks
    placed in the middle of a 300ms silence on either side. 
    
    The first tone is assumed to be 10 kHz, and the last tone is assumed 
    to be 92 kHz.

    Parameters:
    
        mic_rec : 1xnsample np.array with 82 pure tones. 
        
        fs : integer. sampling rate in Hz. Defaults to 192000

    Returns:

        all_pure_tones : 82 x 19200 np.array with all the pure tones. 
                         The pure tones are filtered with a centre frequency +/-
                         2kHz bandpass filter.
    
    *IMPORTANT* The recording  is assumed to be time synchronised with the playback 
                  sequence. By time synchronised I mean that the 
                  whole recording can be broken evenly into 700 ms chunks 
                  with the pure tone occupying exactly the middle 100ms 
                  in each chunk. 
  TODO : 
      1) Extract out the background non playback sections to allow measurement
      of signal to noise ratios 
    '''
    pure_tones = np.arange(10,93)*1000 # the pure tone frequencies
    nyquist_freq = 96000
    pure_tone_size = int(0.1*fs)
    silence_size = int(0.3*fs)

    all_pure_tones = np.zeros((pure_tones.size,19200))
    all_pure_tones_bkg = np.zeros(())
    
    bp_width = 2000 # in Hz
    # the tones start from the 5th chunk 
    for i, freq in enumerate(pure_tones):      
        # take out the audio segment with the pure tone
        tone_start = int((5+i)*0.7*fs)+silence_size
        tone_end = tone_start + pure_tone_size
        pure_tone = mic_rec[tone_start:tone_end]

        bp_cutoffs = np.array([freq-bp_width, freq+bp_width])/nyquist_freq
        b_tone, a_tone = signal.butter(2,bp_cutoffs,                                      
                                       'bandpass')
        padded_tone = np.pad(pure_tone,[1000,1000],'constant',
                             constant_values=[0,0])
        puretone_bp = signal.filtfilt(b_tone, a_tone, padded_tone)
        all_pure_tones[i,:] = puretone_bp[1000:-1000]

    return(all_pure_tones)

def get_sweeps(mic_rec):
    '''Extracts the 5 upward sweeps given a time-aligned recording. 
    *IMPORTANT* : this function assumes that the mic_rec has been aligned
    properly to the playback sequence. 

    Parameters:
        mic_rec : 1 x nsamples np.array of the time aligned mic recording

    Returns:
        all_sweeps : 5 x 134400 samples ( 5 x 0.7 snippets)
    '''
    fs = 192000
    all_sweeps = np.zeros((5, int(0.7*fs)))
    # the tones start from the 5th chunk 
    for i in range(5):      
        # take out the audio segment with the pure tone
        tone_start = int(i*0.7*fs)
        tone_end = tone_start + int(0.7*fs)
        all_sweeps[i,:] = mic_rec[tone_start:tone_end]
    return(all_sweeps)
   
    


def align_mic_rec(mic_rec):
    '''Output a time-aligned recording of a microphone recording. 
    This step is required because the recording started before the 
    playback in the fireface. 
    
    The first sweep is used to find the playback latency. 
    This is the first 700 ms chunk with a 10ms chunk in the middle

    '''
    fs =192000
    pbk_sequence = generate_calibration_playback_sequence()
    one_sweep = pbk_sequence[0][0]
    micrec_chunk = mic_rec[:fs*2]
    nyq = fs/2
    # bandpass to increase cc accuracy 
    b_sweep, a_sweep = signal.butter(2, np.array([10000, 95000])/nyq,
                                        'bandpass')
    bp_micrechunk = signal.filtfilt(b_sweep, a_sweep, micrec_chunk)
    
    # correlate the actual playback with the mic recording
    cc = signal.correlate(bp_micrechunk, one_sweep, 'same')

    # find the delay:
    pbk_delay = np.argmax(cc)
    aligned_micrec = mic_rec[pbk_delay-int(0.35*fs):]
    
    return(pbk_delay, aligned_micrec)
    
    
        
def generate_calibration_playback_sequence():
    '''Code copied from the microphone_calib_playbacks script used 
    on 2018-07-16 and 2019-01-10
    '''
    
    fs = 192000

    # define a common pl.ayback sounds length:
    common_length = 0.7
    numsamples_comlength = int(fs*common_length)

    # Create the calibration chirp : 
    chirp_durn = 0.010
    t_chirp = np.linspace(0,chirp_durn, int(fs*chirp_durn))
    start_f, end_f = 15000, 95000
    sweep = signal.chirp(t_chirp,start_f, t_chirp[-1], end_f, 'linear')
    sweep *= signal.tukey(sweep.size, 0.9)
    sweep *= 0.5

    silence_durn = 0.3
    silence_samples = int(silence_durn*fs)
    silence = np.zeros(silence_samples)
    # create 5 sweeps with silences before & after them :
    sweep_w_leftsilence = np.concatenate((silence,sweep))
    numsamples_to_add = numsamples_comlength - sweep_w_leftsilence.size
    sweep_w_silences = np.concatenate((sweep_w_leftsilence,np.zeros(numsamples_to_add)))
    sweep_w_silences = np.float32(sweep_w_silences)

    all_sweeps = []
    #make a set of 5 sweeps in a row =
    for i in range(5):
        all_sweeps.append(sweep_w_silences)

    # create a set of sinusoidal pulses :
    pulse_durn = 0.1
    pulse_samples = int(fs*pulse_durn)
    t_pulse = np.linspace(0,pulse_durn, pulse_samples)
    pulse_start_f, pulse_end_f  = 10000, 95000
    frequency_step = 1000;
    freqs = np.arange(pulse_start_f, pulse_end_f, frequency_step)

    all_freq_pulses = []
    for each_freq in freqs:
        one_tone = np.sin(2*np.pi*t_pulse*each_freq)
        one_tone *= signal.tukey(one_tone.size, 0.85)

        if each_freq >=80000:
            one_tone *= 0.75
        else:
            one_tone *= 0.5
        one_tone_w_silences = np.float32(np.concatenate((silence,one_tone,silence)))
        all_freq_pulses.append(one_tone_w_silences)

    # setup the speaker playbacks to first play the sweeps and then 
    # the pulses : 
    playback_sounds = [all_sweeps, all_freq_pulses]
    return(playback_sounds)

def calc_rms(X, choose_middle = False, **kwargs):
    '''
    If choose middle is true, then a small section of 1920 samples (10ms) around the 
    middle of X is chosen. 
    '''
    if choose_middle:
        mid_X = int(X.size/2)
        X = X[mid_X-961:mid_X+960]

    sq_X = X**2.0
    mean_sq = np.mean(sq_X)      
    root_msq = np.sqrt(mean_sq)
    return(root_msq)

def calc_peak(X):
    '''Takes the 99th percentile, and *NOT* the max to avoid outlier issues. 
    '''
    peak_X = np.percentile(np.abs(X), 99.5)
    return(peak_X)

def dB(X):
    '''Converts X into decibels
    '''
    dB_X = 20*np.log10(X)
    return(dB_X)

def get_pure_tone_levels(all_pure_tones, frequencies=np.arange(10,93)*1000):
    ''' Calculates the dB peak and rms of the pure tone snippets

    Parameters:
        all_pure_tones : Nfrequencies x nsamples np.array. 

        frequencies : frequencies of the pure tone audio snippets np.array.
                      Defaults to the recorded frequencies.

    Returns:
        pure_tone_levels : Nfrequencies x 3 np.array. 
                          Column 0 : frequencies
                          Column 1 : dB peak
                          Column 2 : dB rms

    TODO : 
        1) Calculate SIGNAL TO NOISE RATIO FOR EACH TONE -- THIS WILL HELP 
        PREVENT STUPID MEASUREMENTS
    '''
    pure_tone_levels = np.zeros((frequencies.size,3))

    pure_tone_levels[:,0] = frequencies
    pure_tone_levels[:,1] = dB(np.apply_along_axis(calc_peak, 1, all_pure_tones))
    pure_tone_levels[:,2] = dB(np.apply_along_axis(calc_rms, 1, all_pure_tones, True))


    return(pure_tone_levels)
    
def make_freq_vs_levels(file_path):
    '''Accepts a file path and outputs pure tone frequencies and levels in dBpeak and 
    dB rms
    Parameters:

        file_path : string. Path to the microphone recording.
    
    Returns:
        pure_tone_levels : Nfreqs x 3 np.array. See get_pure_tone_levels

        all_tones : Nfreqs x  np.array. 

    '''
    rec, fs = sf.read(file_path)
    _, aligned_rec = align_mic_rec(rec)
    all_tones = get_pure_tone_snippets(aligned_rec)
    tone_levels = get_pure_tone_levels(all_tones)
    # check if the output makes sense
    check_3dB_difference(tone_levels)
    
    return(tone_levels, all_tones)

def check_3dB_difference(pure_tone_levels, tolerance = 0.5):
    '''There is an expected 3dB difference between the dBpeak and dBrms for 
    a sine wave. This function checks for the approximate match. 

    Parameters:

        pure_tone_levels: Nfreq x 3 np.array. See get_pure_tone_levels

        tolerance : float >0 . Margin within 3 dB that the differences should
                    lie between. Defaults to 0.5. This default value is an 
                    arbitrary decision!! 

    Note:
        If 3dB_match is False, then a non-fatal warning is issued with the 
        frequency of the snippet that shows anomalous behaviour. 
    '''
    
    dB_diff = pure_tone_levels[:,1] - pure_tone_levels[:,2]

    within_range = np.abs(dB_diff) <= 3 + tolerance
    all_within_range = np.all(within_range)
    if not all_within_range:
        # identify which frequency snippets don't show the expected 3 db difference
        not_within_range = np.where(within_range==False)
        not_within_freqs = pure_tone_levels[not_within_range,0]
        warnings.warn('There are some tones that do not show the expected 3dB difference')
        print(not_within_freqs)
        print(dB_diff[not_within_range])

norm_to_max = lambda X: X - np.max(X)


def calc_mic_frequency_response(mic_tone_levels, refmic_tone_levels,
                       compfilter_size=512, fs=192000):
    '''Makes the microphone Frequency Compensation filter. 

    1. The reference mic is assumed to have a flat frequency response, and thus 
    records the true relative intensities of the pure tones. 
    2. These relative  intensities are compared with the relative intensities of the target microphone. 
    3. The required amplification/attenuation at each tone frequency is calculated
    4. A FIR filter is made with the compensatory response obtained from 3. 

    Parameters:
        mic_tone_levels : Nfreqs x 3. Measurements for the target mic. see get_pure_tone_levels
        
        refmic_tone_levels : Nfreqs x 3. Measurements for the reference mic. see get_pure_tone_levels

        compfilter_size : integer. Size of the compensatory filter. Defaults to 512 samples.

        fs : integer. Sampling rate in Hz. Defaults to 192kHz.

    Return: 

        comp_tone_levels : Nfreqs x 2 np.array. the dB to be added to the target microphone pure tone levels
                           to get a flat frequency response
            
        fc_filter : 1 x compfilter_size np.array. The frequency compensation filter to restore a flat frequency 
                    response for an arbitrary recording from the target microphone. 
   
    '''
    # get relative frequency responses, use dB peak
    refmic_freqresp = norm_to_max(refmic_tone_levels[:,1])
    mic_freqresp = norm_to_max(mic_tone_levels[:,1])

    # get compensatory frequency response in dB 
    comp_response = refmic_freqresp - mic_freqresp
    comp_tone_levels = np.column_stack((mic_tone_levels[:,0], comp_response))
    # make filter:
    freqs = mic_tone_levels[:,0]
    freq_nyq = fs/2
    
    filter_freqs, filter_gains = make_filter_response(comp_response, freqs, freq_nyq)
    fc_filter = signal.firwin2(compfilter_size, filter_freqs, filter_gains,
                               nyq=freq_nyq)
    return(comp_tone_levels, fc_filter)
    
def make_filter_response(comp_gain, tone_freqs, nyquist):
    '''
    '''
    add_freqs_left = np.arange(0,tone_freqs[0],1000)
    add_freqs_right = np.arange(tone_freqs[-1]+1000, nyquist+1000, 1000)
    
    filter_freqs = np.hstack((add_freqs_left, tone_freqs,
                              add_freqs_right))
    filter_gain = np.hstack((np.zeros(add_freqs_left.size),10**(comp_gain/20),
                             np.zeros(add_freqs_right.size)))
    return(filter_freqs, filter_gain)



    



if __name__ == '__main__':
    
    #path = 'H:\\fieldwork_2018_001\\mic_calibrations\\2018-07-16\\'
    path = 'H:\\fieldwork_2018_001\\mic_calibrations\\2019-01-10\\myotis_CPN_mics\\SANKEN_12_1546\\'

    # load a SANKEN recording and get pure tone levels
    sanken_tonelevels, sanken = make_freq_vs_levels(path+'SANKEN_12_1546_aindB_30_azimuth_angle_80_2019-01-11_10-05-51.wav')
    gras_tonelevels, gras = make_freq_vs_levels(path+'GRAS_gaindB_70_azimuth_angle_0_2019-01-11_10-39-31.wav')
    
    ctone_levels, fc = calc_mic_frequency_response(sanken_tonelevels, gras_tonelevels)
    
    comp_sanken_tonelevels = sanken_tonelevels.copy()
    comp_sanken_tonelevels[:,1] += ctone_levels[:,1]
    comp_sanken_tonelevels[:,2] += ctone_levels[:,1]
    
    









