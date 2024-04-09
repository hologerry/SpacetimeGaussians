import os

start_time = 50
duration = 50
time_step = 10

for i in range(start_time, start_time + duration, time_step):
    print(i)
    time_stamp = (i - start_time) / duration
    print(time_stamp)
