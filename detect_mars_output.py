# Tot. PSD completed, relatively fine

import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import scipy
from scipy.signal import stft
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset

def process_data(file_name, time_bins_top3, sum_psd_excluding_top3_amp_atten, on_off_psd, average_low_value, range_low_values, tr_time, tr_times_filt):
    trigger_on_times = []
    trigger_off_times = []

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(time_bins_top3, sum_psd_excluding_top3_amp_atten, label='Sum of PSD values - normalized against min. values')
    ax2.set_xlim([min(tr_times_filt), max(tr_times_filt)])

    for j in range(len(on_off_psd)):
        triggers_psd = on_off_psd[j]
        
        if triggers_psd[0] < len(time_bins_top3):
            index = round((time_bins_top3[triggers_psd[0]] * len(sum_psd_excluding_top3_amp_atten)) / tr_time)

            if abs(time_bins_top3[triggers_psd[1]] - time_bins_top3[triggers_psd[0]]) < 100:
                if (index < len(sum_psd_excluding_top3_amp_atten) and 
                    sum_psd_excluding_top3_amp_atten[index] > (average_low_value + range_low_values) and 
                    abs(time_bins_top3[triggers_psd[1]] - time_bins_top3[triggers_psd[0]]) > 40):
                    
                    ax2.axvline(x=time_bins_top3[triggers_psd[0]], color='red', label='Trig. On' if len(trigger_on_times) == 0 else "")
                    trigger_on_times.append(time_bins_top3[triggers_psd[0]])

                    ax2.axvline(x=time_bins_top3[triggers_psd[1]], color='cyan', label='Trig. Off' if len(trigger_off_times) == 0 else "")
                    trigger_off_times.append(time_bins_top3[triggers_psd[1]])

            else:
                ax2.axvline(x=time_bins_top3[triggers_psd[0]], color='red', label='Trig. On' if len(trigger_on_times) == 0 else "")
                trigger_on_times.append(time_bins_top3[triggers_psd[0]])

                ax2.axvline(x=time_bins_top3[triggers_psd[1]], color='cyan', label='Trig. Off' if len(trigger_off_times) == 0 else "")
                trigger_off_times.append(time_bins_top3[triggers_psd[1]])

    return trigger_on_times, trigger_off_times

#Get data------------------------------------------------------------------------------------
cat_directory = './space_apps_2024_seismic_detection/data/mars/test/catalogs/'
cat_file = cat_directory + 'mars_test_catalogs.csv'
cat = pd.read_csv(cat_file)
cat

results = []
for l in range(len(cat)):

    #Pick time (absolute time)---------------------------------------------------------------------------------------------
    row = cat.iloc[l]           #modify number to choose event (6 - first detection)
    test_filename = row.filename
    # Read miniseed file corresponding to that detection----------------------------------------------
    data_directory = './space_apps_2024_seismic_detection/data/mars/test/data/'
    mseed_file = f'{data_directory}{test_filename}.mseed'
    st = read(mseed_file)

    # This is how you get the data and the time, which is in seconds
    tr = st.traces[0].copy()
    tr_times = tr.times()
    tr_data = tr.data
    starttime = tr.stats.starttime.datetime


#========================================================================================================================
#-------------------------------------------------------Processing--------------------------------------------------------

    # Bandpass filter
    # Set the minimum frequency--------------------------------------------------
    minfreq = 0.75   #ori. 0.75
    maxfreq = 1   #ori. 1.0
    # Going to create a separate trace for the filter data
    st_filt = st.copy()
    st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
    tr_filt = st_filt.traces[0].copy()
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    # Sampling frequency of our trace
    df = tr.stats.sampling_rate

    #STFT
    tr_time = round(max(tr_times_filt))  
    fs = 1     # Sampling frequency in Hz
    t = np.linspace(0, tr_time, fs * tr_time, endpoint=False)
    tr_data = tr_data_filt  
    f, t_stft, Zxx = stft(tr_data, fs=fs)
    # Calculate PSD
    psd = np.abs(Zxx)**2
    psd_sum = np.sum(psd, axis=0)

    # Normalize psd_sum against its minimum
    psd_min = np.min(psd_sum)
    psd_max = np.max(psd_sum)
    sum_psd_normalized_top3 = (psd_sum - psd_min) / (psd_max - psd_min)

    # Define a threshold to identify peaks
    peak_threshold = 0.2
    low_values_mask =  sum_psd_normalized_top3 < peak_threshold
    low_values = sum_psd_normalized_top3[low_values_mask]
    # Calculate the average values
    average_low_value = np.mean(low_values) if len(low_values) > 0 else 0.0 
    highof_low_values_mask = low_values > average_low_value
    highof_low_values = low_values[highof_low_values_mask]
    range_low_values = np.mean(highof_low_values) - average_low_value

    # Define a threshold and an amplification/attenuation factor
    threshold = average_low_value + range_low_values 
    amplification_factor = 2.3  
    # Amplify/attenuate values
    sum_psd_excluding_top3_amp_atten1 = np.where(
        sum_psd_normalized_top3 > threshold,
        sum_psd_normalized_top3 * amplification_factor,
        sum_psd_normalized_top3
    )
    sum_psd_excluding_top3_amp_atten = np.where(
        sum_psd_excluding_top3_amp_atten1 < threshold,
        sum_psd_excluding_top3_amp_atten1 * 1/amplification_factor,
        sum_psd_excluding_top3_amp_atten1
    )

    time_bins_top3 = np.linspace(0, tr_time, len(sum_psd_excluding_top3_amp_atten), endpoint=False)

    thr_on_psd = average_low_value + range_low_values 
    thr_off_psd = average_low_value + 0.1*range_low_values   
    on_off_psd = np.array(trigger_onset(sum_psd_excluding_top3_amp_atten,thr_on_psd,thr_off_psd))



#plt.show()
    all_trigger_times = {}
    # Collecting detection times
    for k in range(len(cat)):
        #time_bins_top3, sum_psd_excluding_top3_amp_atten, on_off_psd, average_low_value, range_low_values, tr_time, tr_times_filt = load_data(k)  # Implement load_data function
        trigger_on_times, trigger_off_times = process_data(k, time_bins_top3, sum_psd_excluding_top3_amp_atten, on_off_psd, average_low_value, range_low_values, tr_time, tr_times_filt)
        all_trigger_times[k] = {
            'trigger_on_times': trigger_on_times,
            'trigger_off_times': trigger_off_times
        }

# Append results to the list
    results.append({
        'filename': test_filename,
        'detect_time_rel(sec)': trigger_on_times,
        'end_detect_time_rel(sec)': trigger_off_times
    })

# Compile dataframe of detections
detect_df = pd.DataFrame(results)

# Output detections to CSV file
detect_df.to_csv('./space_apps_2024_seismic_detection/data/mars/detection_results_mars_test_Insight.csv', index=False)



