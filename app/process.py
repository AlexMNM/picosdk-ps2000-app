# Import modules
from time import time_ns, strftime
from collections import Counter
from itertools import takewhile
from enum import IntEnum
from operator import itemgetter
import json
import os
import logging

# Import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, find_peaks, peak_widths, peak_prominences, resample, decimate
from scipy import ndimage, fft


logger = logging.getLogger(__name__)

# reimplement this because the other one only takes ctypes
def adc_to_mv(values, range_, bitness=16):
    v_ranges = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000]

    return [(x * v_ranges[range_]) / (2**(bitness - 1) - 1) for x in values]

def determine_time_unit(interval_ns):
    unit = 0
    units = ['ns', 'us', 'ms', 's']

    while interval_ns > 5_000:
        interval_ns /= 1000
        unit += 1

    return interval_ns, units[unit]

def seconds_to_samples(time, sample_period=500, sample_units=1e-9):
    return int( time / ( sample_period * sample_units ) )

def samples_to_seconds(samples, sample_period=500, sample_units=1e-9):
    return samples * sample_period * sample_units

def norm_to_mv(normed_val, mean, std):
    return normed_val * std + mean

def mv_to_norm(mv_val, mean, std):
    return (mv_val - mean) / std

def resize_fast(arr):
    re = fft.prev_fast_len(len(arr))
    return arr[:re-1], re

def normalize(signal):
    mean = signal.mean()
    std= signal.std()
    return (signal - mean)/std, mean, std

class Duration:
    def __init__(self, t, unit_exp=-3): # default unit: milliseconds
        self.ns = int(t * 10**(9+unit_exp))
        self.mus = self.ns*1e-3
        self.ms = self.ns*1e-6
        self.s = self.ns*1e-9
class Wave:
    def __init__(self, y, time, t_exp=-9):
        if isinstance(y, list):
            self.y = np.array(y)
        elif isinstance(y, np.ndarray | np.generic):
            self.y = y
        self.n = self.y.size
        if isinstance(time, float | int):
            self.x = np.linspace(0, (self.n-1)*time, self.n)
        elif isinstance(time, list):
            self.x = np.array(time)
        elif isinstance(time, np.ndarray | np.generic):
            self.x = time
        self.tx = t_exp
        self.dt = Duration(self.x[1], unit_exp=self.tx)
        self.interval = Duration(self.x[-1], unit_exp=self.tx)
        


filename = input("Drag file here:")
with open(filename, 'r') as f:
    data = json.load(f)

decimate_factor = 10

dt = data["setup"]["sample_interval"]
a_wave = Wave(decimate(resize_fast(data["raw_data"]["signal_A"])[0], decimate_factor, zero_phase=True), dt * decimate_factor)
b_wave = Wave(decimate(resize_fast(data["raw_data"]["signal_B"])[0], decimate_factor, zero_phase=True), dt * decimate_factor)

A_filtrd = a_wave.y
B_filtrd = b_wave.y

nsamples = a_wave.n
interval = a_wave.interval

# Normalize data
A_norm, A_mean, A_std = normalize(A_filtrd)
B_norm, B_mean, B_std = normalize(B_filtrd)

# Frequency and period
fft_A = np.fft.rfft(A_norm, norm="ortho")
freq_A = abs(fft_A).argmax() * interval.s
period_A = 1 / freq_A

fft_B = np.fft.rfft(B_norm, norm="ortho")
freq_B = abs(fft_A).argmax() * interval.s
period_B = 1 / freq_B

print(f'Recovered frequency: A {freq_A} Hz, B {freq_B} Hz')
print(f'Recovered period: A {period_A * 1e3} ms, B {period_B * 1e3} ms')

# Phase
xcorr = correlate(A_norm, B_norm)
dt = np.arange(1-nsamples ,nsamples)
recovered_timeshift = (dt[xcorr.argmax()] * a_wave.dt.s) * (360 / period_A)
print('Recovered offset: {} degrees'.format(recovered_timeshift))








# Statistics
def modes(data):
    freq = Counter(data)
    mostfreq = freq.most_common()
    return list(takewhile(lambda x_f: x_f[1] == mostfreq[0][1], mostfreq))

def extremes(sig):
    return np.max(sig), np.min(sig)

def get_thresholds(data):
    _sorted = ndimage.uniform_filter1d(sorted(data), 50)
    _gradient = np.gradient(_sorted)
    _gradient -= _gradient.mean()
    _gradient /= _gradient.std()
    _gradient[:int(_gradient.size/5)] = _gradient[-int(_gradient.size/5):] = 0
    _lower = _sorted[np.where(_gradient >= 5*_gradient.std())[0][0]]
    _upper = _sorted[np.where(_gradient >= 3*_gradient.std())[0][-1]]
    return _lower, _upper, _gradient

def clip(sig):
    _max, _min = extremes(sig)
    span = _max-_min
    low_thresh, up_thresh, _ = get_thresholds(sig)
    midpoint = np.mean([low_thresh, up_thresh]) 
    span = up_thresh - low_thresh
    return np.clip(sig, low_thresh, up_thresh), midpoint, span

def peaks_valleys(sig, period):
    clipped, mid, span = clip(sig)
    flipped = 2 * mid - clipped
    peaks, _ = find_peaks(clipped, distance=period*0.6, height=(mid, mid + span), prominence=(0.5*span), plateau_size=period*0.25)
    valleys, _ = find_peaks(flipped, distance=period*0.6, height=(mid, mid + span), prominence=(0.5*span), plateau_size=period*0.25)
    return peaks, valleys, clipped, flipped, mid, span


def bounces(sig, period):
    peaks, valleys = peaks_valleys(sig, period)
    pass

'''
A_max = np.max(A_filtrd)
A_min = np.min(A_filtrd)
B_max = np.max(B_filtrd) 
B_min = np.min(B_filtrd)
A_span = A_max - A_min
B_span = B_max - B_min

A_lower_threshold, A_upper_threshold, A_grd = get_thresholds(A_filtrd)
B_lower_threshold, B_upper_threshold, B_grd = get_thresholds(B_filtrd)

A_mid = np.mean([A_lower_threshold, A_upper_threshold])
B_mid = np.mean([B_lower_threshold, B_upper_threshold])

A_clipped = np.clip(A_filtrd, A_lower_threshold, A_upper_threshold)
A_flipped = 2 * A_mid - A_clipped
B_clipped = np.clip(B_filtrd, B_lower_threshold, B_upper_threshold)
B_flipped = 2 * B_mid - B_clipped

# Peaks
A_peaks, _ = find_peaks(A_clipped, distance=seconds_to_samples(period_A * 0.6), height=(A_mid, A_mid + A_span), prominence= np.max(A_filtrd) - A_mid, plateau_size=seconds_to_samples(period_A * 0.25))

'''

A_peaks, A_valleys, A_clipped, A_flipped, A_mid, _ = peaks_valleys(A_filtrd, period_A/a_wave.dt.s)
B_peaks, B_valleys, B_clipped, B_flipped, B_mid, _ = peaks_valleys(B_filtrd, period_A/a_wave.dt.s)
A_lower_threshold = A_clipped.min()
A_upper_threshold = A_clipped.max()
B_lower_threshold = B_clipped.min()
B_upper_threshold = B_clipped.max()

A_nr_peaks = len(A_peaks)
print('Nr. of peaks A: {}'.format(len(A_peaks)))

A_bounces = np.empty((3, A_nr_peaks * 2))
if A_nr_peaks > 0:
    A_peak_widths = peak_widths(A_clipped, A_peaks, rel_height=0.01)
    print('Widths of A [ms]: \n{}'.format(1000 * samples_to_seconds(A_peak_widths[0])))
    A_pk_prominences = peak_prominences(A_clipped, A_peaks)[0]
    A_valley_widths = peak_widths(A_flipped, A_valleys, rel_height=0.01)

    A_bounces[1][0] = np.where(A_filtrd[:int(A_peak_widths[2][0])] > A_lower_threshold)[0][0]
    A_bounces[2][0] = A_peak_widths[2][0]
    for x in range(0, min(A_nr_peaks - 1, len(A_valley_widths))):
        A_bounces[1][2 * x + 1] = A_peak_widths[3][x]
        A_bounces[2][2 * x + 1] = A_valley_widths[2][x]
    for x in range(1, min(A_nr_peaks, len(A_valley_widths) + 1)):
        A_bounces[1][2 * x] = A_valley_widths[3][x - 1]
        A_bounces[2][2 * x] = A_peak_widths[2][x]
    A_bounces[1][A_nr_peaks * 2 - 1] = A_peak_widths[3][A_nr_peaks - 1]
    A_bounces[2][A_nr_peaks * 2 - 1] = int(A_peak_widths[3][A_nr_peaks - 1]) + np.where(A_filtrd[int(A_peak_widths[3][A_nr_peaks - 1]):(int(A_peak_widths[3][A_nr_peaks - 1] + A_peak_widths[0][-1]/2))] > A_lower_threshold)[0][-1]
    for x in range(0, A_nr_peaks * 2):
        A_bounces[0][x] = A_bounces[2][x] - A_bounces[1][x]



B_nr_peaks = len(B_peaks)
print('Nr. of peaks B: {}'.format(len(B_peaks)))

B_bounces = np.empty((3, B_nr_peaks * 2))
if B_nr_peaks > 0:
    B_peak_widths = peak_widths(B_clipped, B_peaks, rel_height=0.01)
    print('Widths of B [ms]: \n{}'.format(1000 * samples_to_seconds(B_peak_widths[0])))
    B_pk_prominences = peak_prominences(B_clipped, B_peaks)[0]
    B_valley_widths = peak_widths(B_flipped, B_valleys, rel_height=0.01)

    B_bounces[1][0] = np.where(B_filtrd[:int(B_peak_widths[2][0])] > B_lower_threshold)[0][0]
    B_bounces[2][0] = B_peak_widths[2][0]
    for x in range(0, min(B_nr_peaks - 1, len(B_valley_widths))):
        B_bounces[1][2 * x + 1] = B_peak_widths[3][x]
        B_bounces[2][2 * x + 1] = B_valley_widths[2][x]
    for x in range(1, min(B_nr_peaks, len(B_valley_widths) + 1)):
        B_bounces[1][2 * x] = B_valley_widths[3][x - 1]
        B_bounces[2][2 * x] = B_peak_widths[2][x]
    B_bounces[1][B_nr_peaks * 2 - 1] = B_peak_widths[3][B_nr_peaks - 1]
    B_bounces[2][B_nr_peaks * 2 - 1] = int(B_peak_widths[3][B_nr_peaks - 1]) + np.where(B_filtrd[int(B_peak_widths[3][B_nr_peaks - 1]):int(B_peak_widths[3][B_nr_peaks - 1] + B_peak_widths[0][-1]/2)] > B_lower_threshold)[0][-1]
    for x in range(0, B_nr_peaks * 2):
        B_bounces[0][x] = B_bounces[2][x] - B_bounces[1][x]

print("A Bounces in samples:")
print(A_bounces)
print("A Bounces in milliseconds:")
print(samples_to_seconds(A_bounces)*1000)

print("B Bounces in samples:")
print(B_bounces)
print("B Bounces in milliseconds:")
print(samples_to_seconds(B_bounces)*1000)



fig, axs = plt.subplots(6) 

_, units = 'ms'
interval = interval.ms
n = 0

axs[n].plot(a_wave.x, A_filtrd)
n += 1
axs[n].plot(a_wave.y)
n += 1

axs[n].set_xlabel('time/{}'.format(units))
axs[n].hlines(A_mid, 0, interval, linestyle='dotted')
for i, (x1, x2, s) in enumerate(zip(A_bounces[1], A_bounces[2], A_bounces[0])):
    s = samples_to_seconds(s) * 1000
    x1 = samples_to_seconds(x1) * 1000
    x2 = samples_to_seconds(x2) * 1000
    axs[n].text(x1, A_mid - 0.2 + 0.3 * (i % 2), '{0:.2f} ms'.format(s), size='small')
    axs[n].hlines(A_mid, x1, x2, color = 'red')
axs[n].hlines(A_upper_threshold, 0, interval, color = 'green')
axs[n].hlines(A_lower_threshold, 0, interval, color = 'green')
axs[n].plot(np.linspace(0, interval, nsamples), A_filtrd)
n += 1


axs[n].set_xlabel('time/{}'.format(units))
axs[n].plot(np.linspace(0, interval, nsamples), A_clipped)
if A_nr_peaks > 0:
    axs[n].plot(samples_to_seconds(A_peaks)*1000, A_clipped[A_peaks], 'x')
    axs[n].plot(samples_to_seconds(A_valleys)*1000, A_clipped[A_valleys], 'x')
    for i, (x1, x2, s) in enumerate(zip(A_bounces[1], A_bounces[2], A_bounces[0])):
        s = samples_to_seconds(s) * 1000
        x1 = samples_to_seconds(x1) * 1000
        x2 = samples_to_seconds(x2) * 1000
        axs[n].text(x1, A_mid - 0.1 + 0.2 * (i % 2), '{0:.2f} ms'.format(s), size='small')
        axs[n].hlines(A_mid, x1, x2, color = 'red')
    for x1, x2 in zip(A_peak_widths[2], A_peak_widths[3]):
        x1 = samples_to_seconds(x1) * 1000
        x2 = samples_to_seconds(x2) * 1000
        axs[n].hlines(A_mid + 0.1, x1, x2, color='orange')
    for x1, x2 in zip(A_valley_widths[2], A_valley_widths[3]):
        x1 = samples_to_seconds(x1) * 1000
        x2 = samples_to_seconds(x2) * 1000
        axs[n].hlines(A_mid - 0.1, x1, x2, color='green') 
n += 1


axs[n].set_xlabel('time/{}'.format(units))
axs[n].hlines(B_mid, 0, interval, linestyle='dotted')
for i, (x1, x2, s) in enumerate(zip(B_bounces[1], B_bounces[2], B_bounces[0])):
    s = samples_to_seconds(s) * 1000
    x1 = samples_to_seconds(x1) * 1000
    x2 = samples_to_seconds(x2) * 1000
    axs[n].text(x1, B_mid - 0.2 + 0.3 * (i % 2), '{0:.2f} ms'.format(s), size='small')
    axs[n].hlines(B_mid, x1, x2, color = 'red')
axs[n].hlines(B_upper_threshold, 0, interval, color = 'green')
axs[n].hlines(B_lower_threshold, 0, interval, color = 'green')
axs[n].plot(b_wave.x, b_wave.y)
n += 1


axs[n].set_xlabel('time/{}'.format(units))
axs[n].plot(np.linspace(0, interval, nsamples), B_clipped)
if B_nr_peaks > 0:
    axs[n].plot(samples_to_seconds(B_peaks)*1000, B_clipped[B_peaks], 'x')
    axs[n].plot(samples_to_seconds(B_valleys)*1000, B_clipped[B_valleys], 'x')
    for i, (x1, x2, s) in enumerate(zip(B_bounces[1], B_bounces[2], B_bounces[0])):
        s = samples_to_seconds(s) * 1000
        x1 = samples_to_seconds(x1) * 1000
        x2 = samples_to_seconds(x2) * 1000
        axs[n].text(x1, B_mid - 0.1 + 0.2 * (i % 2), '{0:.2f} ms'.format(s), size='small')
        axs[n].hlines(B_mid, x1, x2, color = 'red')
    for x1, x2 in zip(B_peak_widths[2], B_peak_widths[3]):
        x1 = samples_to_seconds(x1) * 1000
        x2 = samples_to_seconds(x2) * 1000
        axs[n].hlines(B_mid + 0.1, x1, x2, color='orange')
    for x1, x2 in zip(B_valley_widths[2], B_valley_widths[3]):
        x1 = samples_to_seconds(x1) * 1000
        x2 = samples_to_seconds(x2) * 1000
        axs[n].hlines(B_mid - 0.1, x1, x2, color='green') 
n += 1


""" axs[n].set_xlabel('time/{}'.format(units))
axs[n].hlines(B_mid, 0, interval, color = 'red')
axs[n].hlines(B_upper_threshold, 0, interval, color = 'green')
axs[n].hlines(B_lower_threshold, 0, interval, color = 'green')
axs[n].plot(np.linspace(0, interval, nsamples), B_filtrd)
n += 1


contour_heights = B_clipped[B_peaks] - B_pk_prominences
B_flip_valley_widths = []
B_flip_valley_widths[0:0] = B_valley_widths
B_flip_valley_widths[1] = 2 * B_mid - B_valley_widths[1]
axs[n].set_xlabel('time/{}'.format(units))
axs[n].plot(B_clipped)
axs[n].plot(B_peaks, B_clipped[B_peaks], 'x')
axs[n].plot(B_valleys, B_clipped[B_valleys], 'x')
axs[n].hlines([B_mid + 0.1 for x in B_peak_widths[1]], *B_peak_widths[2:], color='orange')
axs[n].hlines([B_mid - 0.1 for x in B_valley_widths[1]],*B_valley_widths[2:], color='magenta')  # 2 * B_mid - B_clipped
# axs[n].vlines(x=B_peaks, ymin=contour_heights, ymax=B_clipped[B_peaks])
n += 1
"""
""" axs[n].set_xlabel('time/{}'.format(units))
axs[n].plot(A_flipped)
axs[n].plot(A_valleys, A_flipped[A_valleys], 'x')
axs[n].hlines(*A_valley_widths[1:])
n += 1 """

""" axs[n].set_xlabel('time/{}'.format(units))
axs[n].hlines(B_mid, 0, 1000, color = 'red')
axs[n].plot(np.linspace(0, interval, nsamples), B)
n += 1 """

""" axs[n].set_xlabel('time/{}'.format(units))
axs[n].hlines(B_mid, 0, 1000, color = 'red')
axs[n].hlines(B_upper_threshold, 0, 1000, color = 'green')
axs[n].hlines(B_lower_threshold, 0, 1000, color = 'green')
axs[n].plot(np.linspace(0, interval, nsamples), B_filtrd)
n += 1

axs[n].hlines(B_upper_threshold, 0, 2e6, color = 'green')
axs[n].hlines(B_lower_threshold, 0, 2e6, color = 'green')
axs[n].plot(ndimage.uniform_filter1d(sorted(B_filtrd), 50), color='blue')
n += 1

#axs[n].hist(A_filtrd, density=True, bins=1000)
axs[n].plot(B_grd, color='green')
n += 1 """

""" 
axs[n].hlines(A_upper_threshold, 0, 2e6, color = 'green')
axs[n].hlines(A_lower_threshold, 0, 2e6, color = 'green')
axs[n].plot(ndimage.uniform_filter1d(sorted(A_filtrd), 50), color='blue')
n += 1
"""
""" #axs[n].hist(A_filtrd, density=True, bins=1000)
axs[n].plot(A_grd, color='green')
n += 1 """
          

plt.show()





