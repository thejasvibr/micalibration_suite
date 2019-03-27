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

from sklearn.linear_model import LinearRegression

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

    Parameters:

        mic_rec : 1 x nsamples np.array

    Returns : 

        pbk_delay : integer. Number of samples of delay. 

        aligned_micrec : 1 x alignedsamples np.array. the time-aligned version of mic_rec

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

def calculate_referencemic_received_SPL(refmic_tones, refmic_signal_level={'dBSPL_re20muPa':94,
                                                              'dBpe': -75,
                                                              'dBrms': -78}, 
                                                                refmic_dBgain = 70):
    '''Calculates the received sound pressure levels of each tone of a reference
    calibration microphone with flat frequency response and a measured recording
    of a known SPL tone. Here the GRAS 1/4th inch mic is used, and its companion
    calibrator tone at 94 dB SPL is the recorded sound of known sound pressure level. 
    
    TODO : allow calculate_referencemic_received_SPL to accept kwargs for the 
           refmic_signal_level and refmic_dBgain
    
    Parameters:

        refmic_tones : Nfreq x nsamples. Audio snippets w pure tones

        refmic_signal_level : dictionary. 
                Keys:
                    'dBSPL_re20muPa' : float>0. sound pressure level.
                                       Defaults to GRAS calibrator tone at 94 dB SPL 
                    'dBpe' : float <0. peak level in calibration tone recording.
                                       Defaults to -75 dB pe for the Fireface 802 + GRAS 1/4th inch combination
                    'dBrms' : float<0. rms level in calibration tone recording. Defaults to -78
                                       Defaults to -78 dB pe for the Fireface 802 + GRAS 1/4th inch combination

        dBgain : integer. dB gain for all the recordings. Defaults to 70 dB because those were the settings I used.

    Returns:
        sound_pressure_levels: Ntones x 3 np.array with dB SPL re 20muPa.
                               Column 0 has the playback frequency
                               Column 1 has the dB peak equivalent  SPL
                               Column 2 has the dB rms equivalent SPL

    NOTE: ALL recordings are assumed to have the same gain. If this is *NOT* the case, 
    please manually adjust the gains for each playback and then choose a sensible 
    dBgain value. 
   '''

    calib_mic_clip = {}
    calib_mic_clip['clip_dBpe'] = refmic_signal_level['dBSPL_re20muPa'] - refmic_signal_level['dBpe']
    calib_mic_clip['clip_dBrms'] = refmic_signal_level['dBSPL_re20muPa'] - refmic_signal_level['dBrms']
    pure_tone_levels = get_pure_tone_levels(refmic_tones)

    pure_tone_SPL = np.zeros(pure_tone_levels.shape)
    pure_tone_SPL[:,0] = pure_tone_levels[:,0]
    pure_tone_SPL[:,1] = calib_mic_clip['clip_dBpe'] + pure_tone_levels[:,1] - refmic_dBgain
    pure_tone_SPL[:,2] = calib_mic_clip['clip_dBrms'] + pure_tone_levels[:,2] - refmic_dBgain

    return(pure_tone_SPL)
    
def calculate_clip_level(mic_tone_levels, received_SPL, comp_frequency_response,
                             dBgain, **kwargs):
    '''Calculates clip level of the recording system (mic X ADC device).
    This is done with a regression of the known sound pressure levels(Y) and the 
    recording system's registered levels (X in dB peak or dB rms). 
    
    If this regression is a good fit, then the clip level is found by extrapolating
    the X to a value of 0. 


    Parameters:

        mic_tone_levels : Nfrequencies x 3 np.array.
                          Column 0 : the frequency of playback in Hz
                          Column 1 : the registered level in db peak
                          Column 2 : the registered level in db rms

        received_SPL : Nfrequencies x 1 np.array. The received sound pressure level 
                       re 20 muPa. These values are obtained from a calibrated mic.

        comp_frequency_response : Nfrequencies x 1 np.array. The required attenuation
                      or gain to each playback frequency to achieve a flat frequency
                      response. the values are in dB.

        dBgain : integer. The gain in dB used while recording. 
    
    Keyword Arguments:

        residual_threshold : integer. The acceptable range of the regression residuals
                            in dB. eg. if the threshold is 0.5 dB and there are 
                            residuals > 0.5 dB, then a warning is issued. Defaults
                            to 1 dB, an arbitrary value set by me. 

    Returns :

        clip_level : Dictionary with 2 keys. The clip levels in dB SPL of the system. 
                          'dB_peSPL' : sound pressure level in dB peak
                          'dB_rmsSPL': sound pressure level in dB rms

        SPL_regression : dictionary with 2 sklearn regression objects.
                        Keys : 
                            'regression_dBpe'
                            'regression_dBrms'

    '''
    if 'residual_threshold' in kwargs.keys():
        residual_threshold = kwargs['residual_threshold']
    else:
        residual_threshold = 1 

    # compensate the received levels so the mic has a flat frequency response
    comp_mic_tone_levels = mic_tone_levels.copy()
    comp_mic_tone_levels[:,1] += comp_frequency_response 
    comp_mic_tone_levels[:,2] += comp_frequency_response 
    

    # compensate for the gain while recording:
    comp_mic_tone_levels[:,1:] -= dBgain

    # do regression of the received pressure level and the recorded levels
    rec_dBpe = comp_mic_tone_levels[:,1].reshape(-1,1)
    rec_dBrms = comp_mic_tone_levels[:,2].reshape(-1,1)
    SPL = received_SPL.reshape(-1,1)

    clip_level = {}
    SPL_regressions = []
    for key_name, rec_measure in zip(['dB_peSPL','dB_rmsSPL'],[rec_dBpe, rec_dBrms]):
        cliplevel_reg = LinearRegression()
        cliplevel_reg.fit(rec_measure, SPL)

        # check if the residuals of the regression are low-ish. 
        predicted = cliplevel_reg.predict(rec_measure)
        residuals = predicted  - SPL
        within_limit = np.all(residuals <= np.abs(residual_threshold))

        if not within_limit:
            percent_within = np.round( np.sum(np.abs(residuals)<=residual_threshold)/residuals.size, 2)
            msg1 = 'Some residuals are beyond the threshold for the '+ key_name+' regression' +'\n'
            msg2 = str(percent_within)+' of your data is within the '+str(residual_threshold)+'dB threshold. N=' + str(residuals.size)
            warnings.warn(msg1+msg2)

            
            plt.figure()
            plt.subplot(121)
            plt.plot(SPL, residuals,'*')
            plt.hlines(0, np.min(SPL), np.max(SPL))
            plt.grid()
            plt.xlabel('Received level, dB SPL re 20 muPa');plt.ylabel('residuals, dB')
            plt.title('Regression diagnostic : sound pressure level vs residuals for '+key_name)  
            plt.subplot(122)
            plt.hist(residuals)
        
        clip_level[key_name] = int(np.round(cliplevel_reg.predict(np.array(0).reshape(1,1))))
        SPL_regressions.append(cliplevel_reg)
    return(clip_level, SPL_regressions)

def calibrate_microphone(mic_recording, mic_gain, refmic_recording, **kwargs):
    '''Calculates the clip level of the mic+recording system and outputs the compensatory frequency response.

    Parameters:
        mic_recording : path to the microphone recording. The mic recording of the pre-defined playback
                        See generate_calibration_playback_sequence

        mic_gain : float. dB gain used for the recording. 

        refmic_recording : path to the reference mic recording. The recording the pre-defined playback, 
                        with a *calibration* microphone. In this case, the default is taken as
                        as the settings used for the1/4'' GRAS mic w the Fireface 802.

    Keyword Arguments:
        residual_threshold : float. The absolute range in dB within which the residuals
                             should lie. See calculate_clip_level

        refmic_signal_level : dictionary with entries pertaining to the calibration tone
                              recording from the reference microphone.  
                              See calculate_referencemic_received_SPL

        refmic_dBgain : float. The gain used while recording the calibration tone 
                        with the reference microphone. 
                              

    Returns:

        clip_level : Dictionary with 2 entries.
                     See calculate_clip_level

        compensatory_frequency_response: Dictionary with 2 entries.

            
    '''
    # get recorded tone levels and separate the tones themselves
    refmic_tonelevels, refmic_tones = make_freq_vs_levels(refmic_recording)
    mic_tonelevels, mic_tones = make_freq_vs_levels(mic_recording)

    # generate the compensatory frequency response profile and the filter
    comp_profile, comp_filter = calc_mic_frequency_response(mic_tonelevels, refmic_tonelevels)

    # calculate the received SPL of the tones from the reference mic:
    received_SPL = calculate_referencemic_received_SPL(refmic_tones)

    # get the clip level of the mic+ADC system:
    clip_level, cliplevel_regs = calculate_clip_level(mic_tonelevels, received_SPL[:,1],
                                                      comp_profile[:,1], mic_gain, **kwargs)

    compensatory_frequency_response = {'comp_profile':comp_profile,
                                       'comp_filter':comp_filter}

    return(clip_level, compensatory_frequency_response)

def calculate_received_SPL(audio_snippet, dBgain, compensatory_filter, clip_level):
    '''Calculates the received sound pressure level of an audio snippet
    given the compensatory frequency response filter and the clip level of 
    the mic+ADC system. 
    
    '''
    # compensate for frequency response of microphone
    comp_snippet = signal.convolve(audio_snippet, compensatory_filter,'same')
    # account for the gain used while recording
    dB_peak = dB(calc_peak(comp_snippet)) - dBgain
    dB_rms = dB(calc_rms(comp_snippet)) - dBgain
    # calculate received level based on clipping SPL of the mic+ADC system
    rec_peSPL = clip_level['dB_peSPL'] + dB_peak
    rec_rmsSPL = clip_level['dB_rmsSPL'] + dB_rms

    return(rec_peSPL, rec_rmsSPL)

def calc_sweep_params(rec_sweep):
    b,a = signal.butter(2, 8000/96000.0, 'highpass')
    hp_sweep = signal.filtfilt(b,a, rec_sweep)
    peak_val = np.max(np.abs(hp_sweep))
    rms_val = rms(hp_sweep)
    return(peak_val, rms_val)

if __name__ == '__main__':
    
    #path = 'H:\\fieldwork_2018_001\\mic_calibrations\\2018-07-16\\'
    gras_path = 'H:\\fieldwork_2018_001\\mic_calibrations\\2019-01-10\\myotis_CPN_mics\\SANKEN_11_1545\\'
    gras_file = 'GRAS_gaindB_70_azimuth_angle_0_2019-01-11_11-42-59.wav'

    mic_folder =  'H:\\fieldwork_2018_001\\mic_calibrations\\2019-01-10\\myotis_CPN_mics\\'+'SANKEN_11_1545\\'
    mic_file = 'SANKEN_11_1545_gaindB_30_azimuth_angle_0_2019-01-11_11-29-23.wav'
    # load a SANKEN recording and get pure tone levels
    sanken_tonelevels, sanken = make_freq_vs_levels(mic_folder+mic_file)
    gras_tonelevels, gras = make_freq_vs_levels(gras_path+gras_file)

    ctone_levels, fc = calc_mic_frequency_response(sanken_tonelevels, gras_tonelevels)
    
    comp_sanken_tonelevels = sanken_tonelevels.copy()
    comp_sanken_tonelevels[:,1] += ctone_levels[:,1]
    comp_sanken_tonelevels[:,2] += ctone_levels[:,1]

    tone_SPL = calculate_referencemic_received_SPL(gras)

    clip_levels, clip_regs = calculate_clip_level(sanken_tonelevels, tone_SPL[:,1], ctone_levels[:,1], 30,
    residual_threshold=1)    

    a, b = calibrate_microphone(mic_folder+mic_file, 30, gras_path+gras_file)
    print(a)
    
    rec, fs = sf.read(mic_folder+mic_file)
    _, aligned_rec = align_mic_rec(rec)
    sweeps = get_sweeps(aligned_rec)
    # take out only the relevant sections of the sweep.
    only_sweeps = sweeps[:,57600:57600+1921]

    # get received level of the sweeps:
    rec_peSPL, rec_rmsSPL = calculate_received_SPL(only_sweeps[4,:],30, b['comp_filter'], a);    print(rec_peSPL, rec_rmsSPL)





