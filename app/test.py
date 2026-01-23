from process import Wave, filter_signal, phase_shift, bounces, channel_ax, peaks_valleys, duty_cycle
import json
import matplotlib.pyplot as plt


filename = input("Drag file here:")
with open(filename, 'r') as f:
    data = json.load(f)


dt = data["setup"]["sample_interval"]
a_wave = Wave(*filter_signal(data["raw_data"]["signal_A"], dt))
b_wave = Wave(*filter_signal(data["raw_data"]["signal_B"], dt))


recovered_timeshift, freq_A, period_A, freq_B, period_B = phase_shift(a_wave, b_wave)
print('Recovered offset: {} degrees'.format(recovered_timeshift))


A_pk, A_vly, A_clipped, A_flipped, A_mid, _ = peaks_valleys(a_wave.y, period_A/a_wave.dt.s)
A_bounces = bounces(A_pk, A_vly, A_clipped)
A_duty = duty_cycle(period_A, A_pk[1]['widths'], a_wave.dt.s)

B_pk, B_vly, B_clipped, B_flipped, B_mid, _ = peaks_valleys(b_wave.y, period_B/b_wave.dt.s)
B_bounces = bounces(B_pk, B_vly, B_clipped)
B_duty = duty_cycle(period_B, B_pk[1]['widths'], b_wave.dt.s)


print('Channel A frequency: {} Hz, period: {}, duty cycle: {} %'.format(freq_A, period_A, A_duty))
print('Nr. of peaks A: {}'.format( len(A_pk[0])))
print("A Bounces in samples:")
print(A_bounces[0])
print("A Bounces in milliseconds:")
print(A_bounces[0]*a_wave.dt.ms)

print('Channel B frequency: {} Hz, period: {}, duty cycle: {} %'.format(freq_B, period_B, B_duty))
print('Nr. of peaks B: {}'.format( len(B_pk[0])))
print("B Bounces in samples:")
print(B_bounces[0])
print("B Bounces in milliseconds:")
print(B_bounces[0]*b_wave.dt.ms)


fig, axs = plt.subplots(4) 
fig.set_size_inches(12, 8)
units = 'ms'

axs[-1].set_xlabel('time/{}'.format(units))

channel_ax(axs, 'A', a_wave, A_mid, A_bounces, A_clipped, A_pk, A_vly)
channel_ax(axs[2:], 'B', b_wave, B_mid, B_bounces, B_clipped, B_pk, B_vly)


plt.show()