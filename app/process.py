# Import modules
from collections import Counter
from itertools import takewhile
import json
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

def filter_signal(signal, time_increment):
    D_F = 10
    return decimate(resize_fast(signal)[0], D_F, zero_phase=True), time_increment * D_F
class Duration:
    def __init__(self, t, unit_exp=-3): # default unit: milliseconds
        self.ns = int(t * 10**(9+unit_exp))
        self.mus = self.ns*1e-3
        self.ms = self.ns*1e-6
        self.s = self.ns*1e-9
class Wave:
    def __init__(self, y, dt, t_exp=-9):
        if isinstance(y, list):
            self.y = np.array(y)
        elif isinstance(y, np.ndarray | np.generic):
            self.y = y

        self.n = self.y.size
 
        self._dt = dt
        self._xp = t_exp
        self.dt = Duration(self._dt, unit_exp=self._xp)
        self.x = np.linspace(0, (self.n-1) * self.dt.ms, self.n )
        self.interval = Duration(self.dt.ms * self.n)

def get_freq(sig, duration):
    fft = np.fft.rfft(sig, norm="ortho")
    freq = abs(fft).argmax() / duration
    per = 1 / freq
    return freq, per

def phase_shift(wave_a, wave_b):
    norm_a, _, _ = normalize(wave_a.y)
    norm_b, _, _ = normalize(wave_b.y)
    freq_a, per_a = get_freq(norm_a, wave_a.interval.s)
    freq_b, per_b = get_freq(norm_b, wave_b.interval.s)

    xcorr = correlate(norm_a, norm_b)
    dt = np.arange(1-wave_a.n ,wave_a.n)
    t_shift = (dt[xcorr.argmax()] * wave_a.dt.s) * (360 / per_a)

    return t_shift, freq_a, per_a, freq_b, per_b

def modes(data):
    freq = Counter(data)
    mostfreq = freq.most_common()
    return list(takewhile(lambda x_f: x_f[1] == mostfreq[0][1], mostfreq))

def extremes(sig):
    return np.max(sig), np.min(sig)

def get_thresholds(data):
    _sorted = ndimage.uniform_filter1d(sorted(data), 50)
    _gradient = np.gradient(_sorted)
    #_gradient -= _gradient.mean()
    #_gradient /= _gradient.std()
    _gradient[:int(_gradient.size/5)] = _gradient[-int(_gradient.size/5):] = _gradient.min()
    _lower = _sorted[np.where(_gradient >= np.percentile(_gradient, 98.8))[0][0]]   # 6*_gradient.std()
    _upper = _sorted[np.where(_gradient >= np.percentile(_gradient, 99.8))[0][-1]]  # 6*_gradient.std()
    _lower = 1000 #max(900, _lower)
    _upper = 9000 #min(9100, _upper)
    return _lower, _upper, _gradient

def clip(sig):
    _max, _min = extremes(sig)
    span = _max-_min
    low_thresh, up_thresh, _ = get_thresholds(sig)
    midpoint = np.mean([low_thresh, up_thresh]) 
    span = up_thresh - low_thresh
    return np.clip(sig, low_thresh, up_thresh), midpoint, span

def peaks_valleys(sig, period):
    R_H = 0.01
    clipped, mid, span = clip(sig)
    flipped = 2 * mid - clipped
    p_res = find_peaks(clipped, distance=period*0.4, height=mid, prominence=0.4*span, plateau_size=period*0.2, width=period*0.2, rel_height=R_H)
    v_res = find_peaks(flipped, height=mid, prominence=0.2*span, plateau_size=period*0.2, width=period*0.2, rel_height=R_H)
    return p_res, v_res, clipped, flipped, mid, span

def bounces(p, v, clipped):
    peaks, p_prop = p
    valleys, v_prop = v
    p_cnt = len(peaks)
    v_cnt = len(valleys)
    b_cnt = 2 * p_cnt
    bounces = (np.zeros(b_cnt),np.zeros(b_cnt),np.zeros(b_cnt)) # size, start, end
    if p_cnt > 0:
        first_v = valleys[0] if v_cnt > 0 else 0
        v_offs = 0 if peaks[0] < first_v else -1
        over_min = np.flatnonzero(clipped > clipped.min())
        V = 0
        ST = 1
        ND = 2
        for i in range(p_cnt):
            left = 2 * i
            right = 2 * i + 1
            vir = i - v_offs
            vil = vir - 1
            bounces[ST][left] = v_prop['right_ips'][vil] if v_cnt > 0 and vil > -1 else over_min[over_min > p_prop['left_ips'][0] / 2 ][0]
            bounces[ND][left] = p_prop['left_ips'][i]
            bounces[V][left] = bounces[ND][left] - bounces[ST][left]
            bounces[ST][right] = p_prop['right_ips'][i]
            bounces[ND][right] = v_prop['left_ips'][vir] if v_cnt > 0 and vir < v_cnt else over_min[-1]
            bounces[V][right] = bounces[ND][right] - bounces[ST][right]
        
    return bounces

def duty_cycle(period, peak_widths, dt):
    if len(peak_widths) < 2:
        return 0
    avg_width = np.mean(peak_widths) * dt
    duty = (avg_width / period) * 100
    return duty

def channel_ax(axs, chann, wave, mid, bnc, clipped, pk, vly): 
    n = 0
    y = wave.y
    x = wave.x
    dt = wave.dt.ms
    upp = clipped.max()
    low = clipped.min()
    peaks, _pk = pk
    valleys, _vl = vly

    axs[n].set_title('Channel: ' + chann)
    axs[n].set_ylabel('signal/mV')
    axs[n].hlines(mid, 0, x[-1], linestyle='dotted')
    for i, (x1, x2, s) in enumerate(zip(bnc[1], bnc[2], bnc[0])):
        s = s * dt
        x1 = x1 *dt
        x2 = x2 *dt
        axs[n].text(x1, mid * ( 0.9 - 0.1 * ((-1) ** (i % 2))), '{0:.2f} ms'.format(s), size='small')
        axs[n].hlines(mid, x1, x2, color = 'red')
    axs[n].hlines(upp, 0, x[-1], color = 'orange')
    axs[n].hlines(low, 0, x[-1], color = 'green')
    axs[n].plot(x, y)
    n += 1

    axs[n].set_ylabel('signal/mV')
    axs[n].plot(x, clipped)
    if len(peaks) > 0:
        axs[n].plot(peaks * dt, clipped[peaks], 'x')
        axs[n].plot(valleys * dt, clipped[valleys], 'x')
        for i, (x1, x2, s) in enumerate(zip(bnc[1], bnc[2], bnc[0])):
            s = s * dt
            x1 = x1 *dt
            x2 = x2 *dt
            axs[n].text(x1, mid * ( 0.9 - 0.1 * ((-1) ** (i % 2))), '{0:.2f} ms'.format(s), size='small')
            axs[n].hlines(mid, x1, x2, color = 'red')
        for x1, x2 in zip(_pk['left_ips'], _pk['right_ips']):
            x1 = x1 * dt
            x2 = x2 * dt
            axs[n].hlines(mid + 0.1, x1, x2, color='orange')
        for x1, x2 in zip(_vl['left_ips'], _vl['right_ips']):
            x1 = x1 * dt
            x2 = x2 * dt
            axs[n].hlines(mid - 0.1, x1, x2, color='green') 




# filename = input("Drag file here:")
# with open(filename, 'r') as f:
#     data = json.load(f)


# dt = data["setup"]["sample_interval"]
# a_wave = Wave(*filter_signal(data["raw_data"]["signal_A"], dt))
# b_wave = Wave(*filter_signal(data["raw_data"]["signal_B"], dt))


# recovered_timeshift, freq_A, period_A, freq_B, period_B = phase_shift(a_wave, b_wave)
# print('Recovered offset: {} degrees'.format(recovered_timeshift))


# A_pk, A_vly, A_clipped, A_flipped, A_mid, _ = peaks_valleys(a_wave.y, period_A/a_wave.dt.s)
# A_bounces = bounces(A_pk, A_vly, A_clipped)


# B_pk, B_vly, B_clipped, B_flipped, B_mid, _ = peaks_valleys(b_wave.y, period_B/b_wave.dt.s)
# B_bounces = bounces(B_pk, B_vly, B_clipped)


# print('Nr. of peaks A: {}'.format( len(A_pk[0])))
# print("A Bounces in samples:")
# print(A_bounces[0])
# print("A Bounces in milliseconds:")
# print(A_bounces[0]*a_wave.dt.ms)

# print('Nr. of peaks B: {}'.format( len(B_pk[0])))
# print("B Bounces in samples:")
# print(B_bounces[0])
# print("B Bounces in milliseconds:")
# print(B_bounces[0]*b_wave.dt.ms)


# fig, axs = plt.subplots(4) 
# fig.set_size_inches(12, 8)

# units = 'ms'
# axs[-1].set_xlabel('time/{}'.format(units))

# channel_ax(axs, 'A', a_wave, A_mid, B_bounces, A_clipped, A_pk, A_vly)
# channel_ax(axs[2:], 'B', b_wave, B_mid, B_bounces, B_clipped, B_pk, B_vly)






# """ axs[n].set_xlabel('time/{}'.format(units))
# axs[n].hlines(B_mid, 0, interval, color = 'red')
# axs[n].hlines(B_upper_threshold, 0, interval, color = 'green')
# axs[n].hlines(B_lower_threshold, 0, interval, color = 'green')
# axs[n].plot(np.linspace(0, interval, nsamples), B_filtrd)
# n += 1


# contour_heights = B_clipped[B_peaks] - B_pk_prominences
# B_flip_valley_widths = []
# B_flip_valley_widths[0:0] = B_valley_widths
# B_flip_valley_widths[1] = 2 * B_mid - B_valley_widths[1]
# axs[n].set_xlabel('time/{}'.format(units))
# axs[n].plot(B_clipped)
# axs[n].plot(B_peaks, B_clipped[B_peaks], 'x')
# axs[n].plot(B_valleys, B_clipped[B_valleys], 'x')
# axs[n].hlines([B_mid + 0.1 for x in B_peak_widths[1]], *B_peak_widths[2:], color='orange')
# axs[n].hlines([B_mid - 0.1 for x in B_valley_widths[1]],*B_valley_widths[2:], color='magenta')  # 2 * B_mid - B_clipped
# # axs[n].vlines(x=B_peaks, ymin=contour_heights, ymax=B_clipped[B_peaks])
# n += 1
# """
# """ axs[n].set_xlabel('time/{}'.format(units))
# axs[n].plot(A_flipped)
# axs[n].plot(A_valleys, A_flipped[A_valleys], 'x')
# axs[n].hlines(*A_valley_widths[1:])
# n += 1 """

# """ axs[n].set_xlabel('time/{}'.format(units))
# axs[n].hlines(B_mid, 0, 1000, color = 'red')
# axs[n].plot(np.linspace(0, interval, nsamples), B)
# n += 1 """

# """ axs[n].set_xlabel('time/{}'.format(units))
# axs[n].hlines(B_mid, 0, 1000, color = 'red')
# axs[n].hlines(B_upper_threshold, 0, 1000, color = 'green')
# axs[n].hlines(B_lower_threshold, 0, 1000, color = 'green')
# axs[n].plot(np.linspace(0, interval, nsamples), B_filtrd)
# n += 1

# axs[n].hlines(B_upper_threshold, 0, 2e6, color = 'green')
# axs[n].hlines(B_lower_threshold, 0, 2e6, color = 'green')
# axs[n].plot(ndimage.uniform_filter1d(sorted(B_filtrd), 50), color='blue')
# n += 1

# #axs[n].hist(A_filtrd, density=True, bins=1000)
# axs[n].plot(B_grd, color='green')
# n += 1 """

# """ 
# axs[n].hlines(A_upper_threshold, 0, 2e6, color = 'green')
# axs[n].hlines(A_lower_threshold, 0, 2e6, color = 'green')
# axs[n].plot(ndimage.uniform_filter1d(sorted(A_filtrd), 50), color='blue')
# n += 1
# """
# """ #axs[n].hist(A_filtrd, density=True, bins=1000)
# axs[n].plot(A_grd, color='green')
# n += 1 """
          

# plt.show()





