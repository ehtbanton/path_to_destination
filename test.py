import time
import VL53L1X

tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof.open()
tof.start_ranging(1)  # 1 = short, 2 = medium, 3 = long

print("Reading ToF distance... Ctrl+C to stop")
try:
    while True:
        distance_mm = tof.get_distance()
        print(f"Distance: {distance_mm} mm")
        time.sleep(0.2)
except KeyboardInterrupt:
    pass

tof.stop_ranging()
tof.close()
