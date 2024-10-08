# For martian data

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

#Get data------------------------------------------------------------------------------------
cat_directory = './space_apps_2024_seismic_detection/data/mars/training/catalogs/'
cat_file = cat_directory + 'Mars_InSight_training_catalog.csv'
cat = pd.read_csv(cat_file)
cat

#Pick time (absolute time)---------------------------------------------------------------------------------------------
row = cat.iloc[0]           #modify number to choose event (6 - first detection)
test_filename = row.filename
# Read miniseed file corresponding to that detection----------------------------------------------
data_directory = './space_apps_2024_seismic_detection/data/mars/training/data/'
mseed_file = f'{data_directory}{test_filename}.mseed'
st = read(mseed_file)

# Get data & time
tr = st.traces[0].copy()
tr_times = tr.times()
tr_data = tr.data
starttime = tr.stats.starttime.datetime

#========================================================================================================================
#-------------------------------------------------------Processing--------------------------------------------------------

# Bandpass filter
# Set the min/max frequency--------------------------------------------------
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
# How long should the short-term and long-term window be, in seconds?
sta_len = 120
lta_len = 600
# Run Obspy's STA/LTA to obtain a characteristic function
cft = classic_sta_lta(tr_data_filt, int(sta_len * df), int(lta_len * df))
# Play around with the on and off triggers, based on values in the characteristic function
thr_on = 4
thr_off = 1.5
on_off = np.array(trigger_onset(cft, thr_on, thr_off))

#STFT
tr_time = round(max(tr_times_filt))  
fs = 1     # Sampling frequency in Hz
t = np.linspace(0, tr_time, fs * tr_time, endpoint=False)
tr_data = tr_data_filt  # Example signal
# Perform Short-Time Fourier Transform
f, t_stft, Zxx = stft(tr_data, fs=fs)
# Calculate Power Spectral Density (PSD)
psd = np.abs(Zxx)**2
# Sum PSD values across all frequencies for each time bin
psd_sum = np.sum(psd, axis=0)

# Normalize psd_sum against its minimum
psd_min = np.min(psd_sum)
psd_max = np.max(psd_sum)
sum_psd_normalized_top3 = (psd_sum - psd_min) / (psd_max - psd_min)
#time_bins = np.linspace(0, tr_time, len(psd_sum_normalized), endpoint=False)

# Define a threshold to identify peaks 
peak_threshold = 0.2
# Create a mask to exclude peaks
low_values_mask =  sum_psd_normalized_top3 < peak_threshold
low_values = sum_psd_normalized_top3[low_values_mask]
# Calculate the average value of the signal when it is low
average_low_value = np.mean(low_values) if len(low_values) > 0 else 0.0 
highof_low_values_mask = low_values > average_low_value
highof_low_values = low_values[highof_low_values_mask]
range_low_values = np.mean(highof_low_values) - average_low_value

# Define a threshold and an amplification/attenuation factor
threshold = average_low_value + range_low_values # Adjust this value based on your needs 005########################
amplification_factor = 2.3  # Factor to amplify the values above the threshold 2
# Amplify/attenuate values in sum_psd_excluding_top3 that exceed the threshold
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

# Plot on and off triggers
from scipy import signal
from matplotlib import cm
f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

# Plot the time series and spectrogram)
fig = plt.figure(figsize=(10, 10))

ax1 = plt.subplot(2, 1, 1)
for i in np.arange(0,len(on_off)):
    triggers = on_off[i]
    ax1.axvline(x = tr_times[triggers[0]], color='red', label='Trig. On')
    ax1.axvline(x = tr_times[triggers[1]], color='purple', label='Trig. Off')
# Plot seismogram
ax1.plot(tr_times_filt,tr_data_filt,label ="Signal after bandpass filter, quakes detected using STA/LTA")
ax1.set_xlim([min(tr_times_filt),max(tr_times_filt)])
ax1.legend()

ax2 = plt.subplot(2, 1, 2)
ax2.plot(time_bins_top3, sum_psd_excluding_top3_amp_atten, label='Sum of PSD values - normalized against min. values')
ax2.set_xlim([min(tr_times_filt),max(tr_times_filt)])
for j in np.arange(0,len(on_off_psd)):
    triggers_psd = on_off_psd[j]
    index = round((time_bins_top3[triggers_psd[0]]*len(sum_psd_excluding_top3_amp_atten))/tr_time)
    if abs(time_bins_top3[triggers_psd[1]]-time_bins_top3[triggers_psd[0]]) < 100:       #time period for when PSD sum is too short => does not flag
        if sum_psd_excluding_top3_amp_atten[index] > (average_low_value + range_low_values) and abs(time_bins_top3[triggers_psd[1]]-time_bins_top3[triggers_psd[0]]) > 40:
            ax2.axvline(x = time_bins_top3[triggers_psd[0]], color='red', label='Trig. On')
            ax2.axvline(x = time_bins_top3[triggers_psd[1]], color='cyan', label='Trig. Off')
        else:
            pass
    else:
        ax2.axvline(x = time_bins_top3[triggers_psd[0]], color='red', label='Trig. On')
        ax2.axvline(x = time_bins_top3[triggers_psd[1]], color='cyan', label='Trig. Off')
#ax3.set_xlim([min(tr_times_filt),max(tr_times_filt)])
#plt.title('Sum of Normalized Power Spectral Density (PSD) over Time excl. 3')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Sum of PSD values processed')
ax2.legend()
#plt.grid()
#print(time_bins_top3[triggers_psd[0]])
#print(time_bins_top3[triggers_psd[1]])
plt.show()

# File name and start time of trace-----------------------------------------------
fname = row.filename
#starttime = tr.stats.starttime.datetime
starttime_rel = 0
# Iterate through detection times and compile them
detection_times = []
end_detection_times =[]
fnames = []
for k in np.arange(0,len(on_off_psd)):
    triggers_psd = on_off_psd[k]    
    on_time = starttime_rel + time_bins_top3[triggers_psd[0]]
    off_time = starttime_rel + time_bins_top3[triggers_psd[1]]
    detection_times.append(on_time)
    end_detection_times.append(off_time)
    fnames.append(fname)
# Compile dataframe of detections
detect_df = pd.DataFrame(data = {'filename':fnames,'detect_time_rel(sec)':time_bins_top3[triggers_psd[0]], 
                                 'end_detect_time_rel(sec)':time_bins_top3[triggers_psd[1]]} )
detect_df.head()
# Output detections to CSV file.
detect_df.to_csv('./space_apps_2024_seismic_detection/data/mars/test/catalogs/catalog_test.csv', index=False)