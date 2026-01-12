# Import modules
from time import time_ns, strftime
from ctypes import POINTER, c_int16, c_uint32
from collections import Counter
from itertools import takewhile
from enum import IntEnum
import json
import socket
import threading
import os
import logging

# Import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, find_peaks, peak_widths, peak_prominences
from scipy import ndimage

# Import picosdk
from picosdk.ps2000 import ps2000
from picosdk.functions import assert_pico2000_ok
from picosdk.ctypes_wrapper import C_CALLBACK_FUNCTION_FACTORY

# Import process
from process import Wave, filter_signal, phase_shift, bounces, channel_ax, peaks_valleys


logger = logging.getLogger(__name__)

class TriggerDirection(IntEnum) :
    PS2000_RISING = 0
    PS2000_FALLING = 1

CALLBACK = C_CALLBACK_FUNCTION_FACTORY(None, POINTER(POINTER(c_int16)), c_int16, c_uint32, c_int16, c_int16, c_uint32)

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

def handle_request(c, req, b_sock, b_addr): 
        #buffer = c.socket.recv(1024)
       # try:
            #buffer.data 
            

            # Extract settings
            #settings = buffer.data
            #picoDevice.set_trigger(leading_wave= settings.leading_wave)
            #picoDevice.set_samples(expected_pulses= settings.pulses)
            response = ['PICO','AqStarted']
            c.send(json.dumps(response).encode())
            print("Gathering...")

            picoDevice.set_samples(req[2])
            picoDevice.run_streaming()
            valuesA, valuesB, trigger_start = picoDevice.gather()
            picoDevice.stop()

            dt = picoDevice.sample_interval
            a_wave = Wave(*filter_signal(valuesA, dt))
            b_wave = Wave(*filter_signal(valuesB, dt))

            recovered_timeshift, freq_A, period_A, freq_B, period_B = phase_shift(a_wave, b_wave)
            print('Recovered offset: {} degrees'.format(recovered_timeshift))

            A_pk, A_vly, A_clipped, A_flipped, A_mid, _ = peaks_valleys(a_wave.y, period_A/a_wave.dt.s)
            A_bounces = bounces(A_pk, A_vly, A_clipped)


            B_pk, B_vly, B_clipped, B_flipped, B_mid, _ = peaks_valleys(b_wave.y, period_B/b_wave.dt.s)
            B_bounces = bounces(B_pk, B_vly, B_clipped)


            print('Nr. of peaks A: {}'.format( len(A_pk[0])))
            print("A Bounces in samples:")
            print(A_bounces[0])
            print("A Bounces in milliseconds:")
            print(A_bounces[0]*a_wave.dt.ms)

            print('Nr. of peaks B: {}'.format( len(B_pk[0])))
            print("B Bounces in samples:")
            print(B_bounces[0])
            print("B Bounces in milliseconds:")
            print(B_bounces[0]*b_wave.dt.ms)

            # Save values
            data = {
                "setup": { 
                    #"first_edge": settings.leading_wave,
                    #"expected_pulses": settings.pulses,
                    #"expected_period": 1 / settings.expected_pulses,
                    #"expected_pulse_width": 1 / settings.expected_pulses * 0.4,
                    "sample_interval": dt,
                    #"time_interval": 1 + 0.2 * 1 / settings.expected_pulses,
                    "samples": len(valuesA)
                },

                "raw_data": {
                    "signal_A": valuesA,
                    "signal_B": valuesB
                },

                "results": {
                    "A_bounces": (A_bounces[0]*a_wave.dt.ms).tolist(),
                    "B_bounces": (B_bounces[0]*b_wave.dt.ms).tolist(),
                    "A_frequency": freq_A,
                    "B_frequency": freq_B,
                    "A_period": period_A, 
                    "B_period": period_B,
                    "phase_delta":  recovered_timeshift,
                    "A_peak_cnt": len(A_pk[0]),
                    "B_peak_cnt": len(B_pk[0])
                }
            }
            timestamp = strftime("%Y%m%d-%H%M%S")
            filename = "./app/__pycache__/signals/signal_data_" + timestamp + ".json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(data, f)


            del data["raw_data"]
            filename = "D:/appcache/ps2000/result_file" + ".json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(data, f)

            # Send response
            msg = ['PICO', 'AqResults']
            c.send(json.dumps(msg).encode())

            
            global fig, axs
            for ax in axs:
                ax.clear()

            #fig, axs = plt.subplots(4) 
            units = 'ms'
            axs[-1].set_xlabel('time/{}'.format(units))
            channel_ax(axs, 'A', a_wave, A_mid, B_bounces, A_clipped, A_pk, A_vly)
            channel_ax(axs[2:], 'B', b_wave, B_mid, B_bounces, B_clipped, B_pk, B_vly)

            plt.pause(0.01)           
            
            
    
class StreamingDevice:
    def __init__(self, gather_values, sample_interval, potential_range=ps2000.PS2000_VOLTAGE_RANGE['PS2000_1V'], pretrigger = 4000):
        self.device = ps2000.open_unit()
        # signal generator for testing
        #res = ps2000.ps2000_set_sig_gen_built_in(self.device.handle, 1_000_000, 2_000_000, 1, 32, 32, 0, 0, 0, 0)
        #assert_pico2000_ok(res)

        self.potential_range = potential_range
        self.gather_values = gather_values
        self.sample_interval = sample_interval
        self.pretrigger = pretrigger # seconds_to_samples(2e-3) # 2 millisecond pretrigger


        res = ps2000.ps2000_set_channel(self.device.handle, ps2000.PICO_CHANNEL["A"], True, True, potential_range)
        assert_pico2000_ok(res)
        res = ps2000.ps2000_set_channel(self.device.handle, ps2000.PICO_CHANNEL["B"], True, True, potential_range)
        assert_pico2000_ok(res)
        self.set_trigger(leading_wave= 'A')

    def set_trigger(self, leading_wave):
        threshold = int(32_767 / 2) # about half the potential range in ADC values (-32_767 -> +32_767)
        direction = TriggerDirection.PS2000_RISING
        delay = 0 # percent -100% -> +100%
        auto_trigger = 2_000 # milliseconds
        res = ps2000.ps2000_set_trigger(
            self.device.handle, 
            ps2000.PICO_CHANNEL[leading_wave], 
            threshold, 
            direction, 
            delay, 
            auto_trigger)
        assert_pico2000_ok(res)

    def set_samples(self, expected_pulses):
        expected_period = 1 / expected_pulses
        time_interval = 1 + 0.2 * expected_period
        self.samples = seconds_to_samples(time_interval, self.sample_interval)

    def set_pretrigger(self, expected_pulses):
        expected_period = 1 / expected_pulses # seconds
        self.pretrigger = seconds_to_samples(0.5 * expected_period)

    def run_streaming(self):
        # start 'fast-streaming' mode
        res = ps2000.ps2000_run_streaming_ns(
            self.device.handle,
            self.sample_interval,
            ps2000.PS2000_TIME_UNITS['PS2000_NS'], #Units: Nanoseconds
            22_000_000, #100_000, # max_samples
            False,  # auto_stop
            1,  # noOfSamplesPerAggregate
            50_000  # overview_buffer_size
        )
        assert_pico2000_ok(res)

        self.start_time = time_ns()
        self.end_time = time_ns()

    def close(self):
        ps2000.ps2000_stop(self.device.handle)
        self.device.close()

    def gather(self):
        adc_valuesA = []
        adc_valuesB = []
        pretriggerA = []
        pretriggerB = []
        triggered = False
        triggered_at = 0

        def get_overview_buffers(buffers, _overflow, _triggered_at, _triggered, _auto_stop, n_values):
            nonlocal triggered
            nonlocal triggered_at

            if not triggered:
                pretriggerA.extend(buffers[0][0:n_values])
                pretriggerB.extend(buffers[2][0:n_values])

            if _triggered:
                triggered = True
                triggered_at = len(pretriggerA) + _triggered_at
                self.start_time = time_ns() + (_triggered_at - self.pretrigger - n_values ) * self.sample_interval

            if triggered:
                adc_valuesA.extend(buffers[0][0:n_values])
                adc_valuesB.extend(buffers[2][0:n_values])

            
        callback = CALLBACK(get_overview_buffers)

        while ((len(adc_valuesA) < self.gather_values) or not triggered):
            ps2000.ps2000_get_streaming_last_values(
                self.device.handle,
                callback
            )

            if len(pretriggerA) > self.pretrigger:
                pretriggerA[0:-self.pretrigger] = []
                pretriggerB[0:-self.pretrigger] = []
            

        adc_valuesA[0:0] = pretriggerA[-self.pretrigger:]
        adc_valuesB[0:0] = pretriggerB[-self.pretrigger:]

        self.end_time = time_ns()

        return adc_to_mv(adc_valuesA, self.potential_range), adc_to_mv(adc_valuesB, self.potential_range), triggered_at

    def stop(self):
        ps2000.ps2000_stop(self.device.handle)



# Setup
#first_edge = 'A' # A or B, depending on the direction of rotation
expected_pulses = 8 # how many pulses should the encoder have in one turn / second
expected_period = 1 / expected_pulses # seconds
expected_pulse_width = expected_period * 0.4 # seconds
sample_interval = 500 # sample interval in nanoseconds
time_interval = 1 + 0.2 * expected_period # time interval for testing, in seconds
samples = seconds_to_samples(time_interval, sample_interval) # how many samples in the time interval
pretrigger = seconds_to_samples(0.5 * expected_period)


# Start device
picoDevice = StreamingDevice(samples, sample_interval, potential_range=ps2000.PS2000_VOLTAGE_RANGE['PS2000_20V'], pretrigger=pretrigger)

# Setup server
bind_ip = '' 
broker_port = 1881
bind_port = 8000
l_server = socket.create_server((bind_ip, bind_port))
print(socket.gethostname())

brocker_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
brocker_addr = ('',broker_port)

# we tell the server to start listening with a maximum backlog of connections set to 5
l_server.listen(5) 
print(f"[+] Listening on port {bind_ip} : {bind_port}")  


plt.show()
plt.ion()
fig, axs = plt.subplots(4) 

# main loop
while True:
    
    c_sock, partner = l_server.accept() 
    print(f"[+] Connection established from: {partner[0]}:{partner[1]} | Socket: {c_sock}")
    print(f"[+] Accepted connection from: {partner[0]}:{partner[1]}")


    
    try: 
        raw = c_sock.recv(2048).decode()
        print(raw)
        request = json.loads(raw)
        print(f"[+] Recieved: {request}")
    except ConnectionResetError: 
        continue

    match request[1]:
            case "StartAq":

                handle_request(c_sock, request, brocker_sock, brocker_addr)  

            case _:
                print("Unknown request received")
                
    c_sock.close()  
    
    #plt.pause(2)
    #Close device
    #picoDevice.close()



