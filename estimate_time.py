def estimate_travel_time(volume, distance_km, speed_kmph=60, intersection_delay=30):
    time_base = distance_km / speed_kmph * 3600
    volume_factor = 1 + (volume / 1000)
    return time_base * volume_factor + intersection_delay
