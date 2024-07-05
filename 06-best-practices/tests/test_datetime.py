from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

print(dt(1, 1), dt(1, 10))
print(dt(1, 2), dt(1, 10))
print(dt(1, 2, 0), dt(1, 2, 59))
print(dt(1, 2, 0), dt(2, 2, 1))