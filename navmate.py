"""
NavMate - Raspberry Pi Navigation Assistant
"""

import os
import VL53L1X
import time
import RPi.GPIO as GPIO
import serial
import pigpio
import sys
import subprocess
import pynmea2
import asyncio
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import datetime
from picamera2 import Picamera2

# --- Portable file paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
LABELMAP_PATH = os.path.join(SCRIPT_DIR, "labelmap.txt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "detect.tflite")
IMAGE_DIR = os.path.join(SCRIPT_DIR, "captured_images")
MESSAGES_FILE = os.path.join(SCRIPT_DIR, "messages.txt")
os.makedirs(IMAGE_DIR, exist_ok=True)

VIBRATION_PIN = 18
BUTTON1_PIN = 23
BUTTON2_PIN = 24
BUTTON3_PIN = 25
PHONE_NUMBER = "+63xxxxxxxxx"
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(VIBRATION_PIN, GPIO.OUT)
GPIO.setup(BUTTON1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON3_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

command_lock = asyncio.Lock()
vl53l1x_lock = asyncio.Lock()  # Lock for distance sensor access
message_file_lock = asyncio.Lock()  # Lock for message file access

# Global settings for toggleable features
vibration_enabled = True
distance_voice_enabled = True

# Message storage functions
async def store_message(message):
    """Store a message in the messages file with timestamp"""
    async with message_file_lock:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(MESSAGES_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
            print(f"Message stored: {message}")
        except Exception as e:
            print(f"Error storing message: {e}")

async def get_last_message():
    """Retrieve the last user message from the messages file (excluding system messages)"""
    async with message_file_lock:
        try:
            if not os.path.exists(MESSAGES_FILE):
                return None
            with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Look for the last actual user message (not system messages)
            for line in reversed(lines):
                line = line.strip()
                if line and "] " in line:
                    message_content = line.split("] ", 1)[1]
                    # Only return guardian messages or regular messages, not system messages
                    if message_content.startswith("[GUARDIAN]") or message_content.startswith("[MESSAGE]"):
                        # Remove the tag and return clean message
                        if message_content.startswith("[GUARDIAN]"):
                            return message_content[10:].strip()  # Remove "[GUARDIAN] "
                        elif message_content.startswith("[MESSAGE]"):
                            return message_content[9:].strip()   # Remove "[MESSAGE] "
            return None
        except Exception as e:
            print(f"Error reading messages: {e}")
            return None

async def get_all_messages():
    """Retrieve all messages from the messages file"""
    async with message_file_lock:
        try:
            if not os.path.exists(MESSAGES_FILE):
                return []
            with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [line.strip() for line in lines if line.strip()]
        except Exception as e:
            print(f"Error reading all messages: {e}")
            return []

async def clear_old_messages(keep_last_n=10):
    """Keep only the last N messages in the file"""
    async with message_file_lock:
        try:
            if not os.path.exists(MESSAGES_FILE):
                return
            with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > keep_last_n:
                with open(MESSAGES_FILE, "w", encoding="utf-8") as f:
                    f.writelines(lines[-keep_last_n:])
                print(f"Cleaned message file, kept last {keep_last_n} messages")
        except Exception as e:
            print(f"Error cleaning messages: {e}")

class VoiceOverPoll:
    def __init__(self):
        self.queue = asyncio.PriorityQueue()
        self.current_process = None
        self.lock = asyncio.Lock()
        self.counter = 0

    async def speak(self, text, priority=10, allow_if_locked=True):
        if not allow_if_locked and command_lock.locked():
            return
        async with self.lock:
            if priority <= 5 and self.current_process and self.current_process.poll() is None:
                self.current_process.terminate()
                await asyncio.sleep(0.05)
            self.counter += 1
            await self.queue.put((priority, self.counter, text))

    async def voice_over_loop(self):
        while True:
            _, _, text = await self.queue.get()
            async with self.lock:
                loop = asyncio.get_event_loop()
                self.current_process = await loop.run_in_executor(
                    None,
                    lambda: subprocess.Popen(
                        ['espeak', '-v', 'en+f3', '-s', '155', '-p', '40', '-a', '200', '--punct=', text],
                        stderr=subprocess.DEVNULL
                    )
                )
            while self.current_process.poll() is None:
                await asyncio.sleep(0.01)
            self.current_process = None

voice_poll = VoiceOverPoll()

async def vibrate_once_async(long_ms=600):
    global vibration_enabled
    if not vibration_enabled:
        return
    GPIO.output(VIBRATION_PIN, GPIO.HIGH)
    await asyncio.sleep(long_ms / 1000)
    GPIO.output(VIBRATION_PIN, GPIO.LOW)

async def vibrate_twice_async(short_ms=200, gap_ms=150):
    global vibration_enabled
    if not vibration_enabled:
        return
    for _ in range(2):
        GPIO.output(VIBRATION_PIN, GPIO.HIGH)
        await asyncio.sleep(short_ms / 1000)
        GPIO.output(VIBRATION_PIN, GPIO.LOW)
        await asyncio.sleep(gap_ms / 1000)

def vibrate_once(long_ms=600):
    GPIO.output(VIBRATION_PIN, GPIO.HIGH)
    time.sleep(long_ms / 1000)
    GPIO.output(VIBRATION_PIN, GPIO.LOW)

def vibrate_twice(short_ms=200, gap_ms=150):
    for _ in range(2):
        GPIO.output(VIBRATION_PIN, GPIO.HIGH)
        time.sleep(short_ms / 1000)
        GPIO.output(VIBRATION_PIN, GPIO.LOW)
        time.sleep(gap_ms / 1000)

GPS_RX = 17
BAUD = 9600
os.system("sudo killall pigpiod")
time.sleep(0.5)
os.system("sudo pigpiod")
time.sleep(0.5)

pi = pigpio.pi()
if not pi.connected:
    print("WARNING: Pigpio not running!")
    sys.exit(1)

pi.bb_serial_read_open(GPS_RX, BAUD)

class LatestGPS:
    def __init__(self):
        self.lat = None
        self.lon = None

latest_gps = LatestGPS()

async def get_gps_fix_async(timeout=3):
    start_time = time.time()
    buffer = b""
    while time.time() - start_time < timeout:
        count, data = pi.bb_serial_read(GPS_RX)
        if count:
            buffer += data
            lines = buffer.split(b'\n')
            buffer = lines[-1]
            for line in lines[:-1]:
                line_str = line.decode('ascii', errors='replace').strip()
                if line_str.startswith('$'):
                    try:
                        msg = pynmea2.parse(line_str)
                        if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                            lat = msg.latitude
                            lon = msg.longitude
                            if (hasattr(msg, 'fix_quality') and int(getattr(msg, 'fix_quality', 0)) > 0) or \
                               (hasattr(msg, 'status') and msg.status == 'A'):
                                return lat, lon
                    except pynmea2.ParseError:
                        continue
        await asyncio.sleep(0.1)
    return None, None

async def gps_background_loop():
    while True:
        lat, lon = await get_gps_fix_async(timeout=2)
        if lat and lon:
            latest_gps.lat = lat
            latest_gps.lon = lon
        await asyncio.sleep(0.5)

sim800l = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

async def initialize_modem():
    """Initialize the SIM800L modem with proper settings"""
    async with command_lock:
        # Clear any pending data
        sim800l.reset_input_buffer()
        sim800l.reset_output_buffer()
        
        # Basic initialization sequence
        sim800l.write(b'AT\r')
        await asyncio.sleep(0.5)
        sim800l.reset_input_buffer()  # Clear echo
        
        # Disable echo to prevent command interference
        sim800l.write(b'ATE0\r')
        await asyncio.sleep(0.5)
        sim800l.reset_input_buffer()
        
        # Set text mode for SMS
        sim800l.write(b'AT+CMGF=1\r')
        await asyncio.sleep(0.5)
        sim800l.reset_input_buffer()
        
        # Set SMS storage to SIM card
        sim800l.write(b'AT+CPMS="SM","SM","SM"\r')
        await asyncio.sleep(0.5)
        sim800l.reset_input_buffer()
        
        # Enable caller ID notification
        sim800l.write(b'AT+CLIP=1\r')
        await asyncio.sleep(0.5)
        sim800l.reset_input_buffer()
        
        # Enable call status notifications
        sim800l.write(b'AT+CRC=1\r')
        await asyncio.sleep(0.5)
        sim800l.reset_input_buffer()
        
        # Set auto-answer to 0 rings (immediate reject)
        sim800l.write(b'ATS0=0\r')
        await asyncio.sleep(0.5)
        sim800l.reset_input_buffer()
        
        print("SIM800L modem initialized successfully with call blocking")

async def read_modem_reply_async(timeout=2):
    end_time = time.time() + timeout
    reply = b""
    while time.time() < end_time:
        if sim800l.in_waiting:
            reply += sim800l.read(sim800l.in_waiting)
        await asyncio.sleep(0.01)
    return reply.decode(errors='ignore').strip()

async def send_sms_with_location(number, gps_lat, gps_lon):
    if isinstance(gps_lat, float) and isinstance(gps_lon, float):
        maps_link = f"https://www.google.com/maps/@{gps_lat},{gps_lon},20z"
        message = f"User sent you an alert\nGPS Location: {gps_lat:.6f}, {gps_lon:.6f}\nView on map: {maps_link}"
    else:
        message = "User sent you an alert\nUnable to get GPS location"
    await send_sms_async(number, message)

async def send_sms_async(number, message):
    async with command_lock:
        # Clear buffers before starting
        sim800l.reset_input_buffer()
        sim800l.reset_output_buffer()
        
        # Ensure modem is responsive
        sim800l.write(b'AT\r')
        reply = await read_modem_reply_async(2)
        if "OK" not in reply:
            sim800l.write(b'AT\r')
            await read_modem_reply_async(2)
        
        # Set text mode
        sim800l.write(b'AT+CMGF=1\r')
        reply = await read_modem_reply_async(2)
        if "OK" not in reply:
            await voice_poll.speak("Error setting SMS mode")
            return
        
        # Start SMS composition
        sim800l.write(f'AT+CMGS="{number}"\r'.encode())
        reply = await read_modem_reply_async(3)
        if '>' not in reply:
            await voice_poll.speak("Error starting SMS")
            return
        
        # Send message content and terminator
        sim800l.write(message.encode() + b"\x1A")
        reply = await read_modem_reply_async(10)
        
        if "OK" in reply or "+CMGS:" in reply:
            await voice_poll.speak("Message sent")
        else:
            await voice_poll.speak("Error sending message")
        
        # Clear buffers after sending
        await asyncio.sleep(0.5)
        sim800l.reset_input_buffer()

async def make_call_async(number):
    async with command_lock:
        sim800l.write(b'AT\r')
        await read_modem_reply_async(2)
        sim800l.write(f'ATD{number};\r'.encode())
        await voice_poll.speak("Calling now")

async def hangup_call_async():
    """Hang up any active call"""
    async with command_lock:
        sim800l.write(b'ATH\r')
        await read_modem_reply_async(2)
        print("Call hung up")

async def block_call_async():
    """Block/reject incoming call immediately"""
    async with command_lock:
        # Send busy signal to reject the call immediately
        sim800l.write(b'AT+CHUP\r')
        await read_modem_reply_async(1)
        # Alternative method - hang up immediately
        sim800l.write(b'ATH\r')
        await read_modem_reply_async(1)
        print("Call blocked/rejected")

def make_call(number):
    asyncio.create_task(make_call_async(number))

tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof.open()
tof.start_ranging(3)

async def safe_get_distance():
    async with vl53l1x_lock:
        return get_safe_distance()

def get_safe_distance():
    try:
        distance = tof.get_distance()
        if 0 < distance < 8000:
            return distance
        elif distance >= 8000:
            return 4000
        else:
            return -1
    except Exception as e:
        print(f"Distance sensor read error: {e}")
        return -1

async def safe_reinitialize_distance_sensor():
    async with vl53l1x_lock:
        return reinitialize_distance_sensor()

def reinitialize_distance_sensor():
    global tof
    try:
        print("Reinitializing distance sensor...")
        tof.stop_ranging()
        time.sleep(0.2)
        tof.start_ranging(3)
        time.sleep(0.3)
        test_distance = tof.get_distance()
        print(f"Distance sensor reinitialized successfully (test read: {test_distance}mm)")
        return True
    except Exception as e:
        print(f"Failed to reinitialize distance sensor: {e}")
        try:
            print("Attempting complete sensor reset...")
            tof.close()
            time.sleep(1.0)
            tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
            tof.open()
            tof.start_ranging(3)
            time.sleep(0.5)
            test_distance = tof.get_distance()
            print(f"Distance sensor reset and reinitialized (test read: {test_distance}mm)")
            return True
        except Exception as reset_error:
            print(f"Complete sensor reset failed: {reset_error}")
            return False

# Portable labelmap and model loading
with open(LABELMAP_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))
picam2.start()

class SystemStatus:
    def __init__(self):
        self.button_states = {"23": False, "24": False, "25": False}
        self.last_display_update = 0
        self.distance = "No reading"
        self.sms_status = "Idle"
        self.detected_object = "None"
        self.last_received_sms = ""
        self.last_received_message = ""  # Stores only the last actual guardian message

    def update_button_state(self, pin, pressed):
        self.button_states[str(pin)] = pressed

    def update_distance(self, distance_cm):
        if distance_cm > 0:
            self.distance = f"{distance_cm:.1f} cm"
        else:
            self.distance = "No reading"

    def update_sms_status(self, status):
        self.sms_status = status

    def update_detected_object(self, obj):
        self.detected_object = obj

    def update_received_sms(self, msg):
        self.last_received_sms = msg

    async def get_last_stored_message(self):
        """Retrieve the last message from file storage"""
        return await get_last_message()

system_status = SystemStatus()

def get_memory_usage():
    meminfo = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            parts = line.split()
            key = parts[0].rstrip(":")
            value = int(parts[1])
            meminfo[key] = value
    total = meminfo.get("MemTotal", 0) // 1024
    free = (meminfo.get("MemFree", 0) + meminfo.get("Buffers", 0) + meminfo.get("Cached", 0)) // 1024
    used = total - free
    return total, used, free

_prev_cpu = None
def get_cpu_usage():
    global _prev_cpu
    with open("/proc/stat", "r") as f:
        line = f.readline()
        if not line.startswith("cpu "):
            return 0.0
        parts = [float(x) for x in line.strip().split()[1:]]
        idle = parts[3] + parts[4]
        total = sum(parts)
    if _prev_cpu is None:
        _prev_cpu = (idle, total)
        return 0.0
    prev_idle, prev_total = _prev_cpu
    idle_delta = idle - prev_idle
    total_delta = total - prev_total
    _prev_cpu = (idle, total)
    if total_delta == 0:
        return 0.0
    usage = 100.0 * (1.0 - idle_delta / total_delta)
    return round(usage, 1)

def get_gpu_info():
    try:
        temp = subprocess.check_output(['vcgencmd', 'measure_temp']).decode()
        mem = subprocess.check_output(['vcgencmd', 'get_mem', 'gpu']).decode()
        return temp.strip(), mem.strip()
    except Exception:
        return "N/A", "N/A"

def display_status():
    print("\033[2J\033[H", end="")
    print(f"Distance: {system_status.distance}")
    lat = latest_gps.lat
    lon = latest_gps.lon
    print(f"GPS Latitude: {lat if lat else 'No fix'}")
    print(f"GPS Longitude: {lon if lon else 'No fix'}")
    print(f"Detected Object: {system_status.detected_object}")
    for pin in ["23", "24", "25"]:
        button_num = "1" if pin == "23" else "2" if pin == "24" else "3"
        state = "PRESSED" if system_status.button_states[pin] else "Released"
        print(f"Button{button_num}: {state}")
    print(f"Process of SMS/Call: {system_status.sms_status}")
    sms_display = system_status.last_received_sms if system_status.last_received_sms else "None"
    print(f"Last Received SMS: {sms_display}")
    print(f"Messages stored in file: {MESSAGES_FILE}")
    print(f"Vibration Motor: {'Enabled' if vibration_enabled else 'Disabled'}")
    print(f"Distance Voice: {'Enabled' if distance_voice_enabled else 'Disabled'}")
    total_mem, used_mem, free_mem = get_memory_usage()
    print(f"Memory Usage: {used_mem}MB used / {total_mem}MB total ({free_mem}MB free)")
    cpu_usage = get_cpu_usage()
    print(f"CPU Usage: {cpu_usage}%")
    gpu_temp, gpu_mem = get_gpu_info()
    print(f"GPU Temp: {gpu_temp}")
    print(f"GPU Mem: {gpu_mem}")
    print("-" * 50)

async def detect_object_non_blocking():
    async with command_lock:
        await detect_object()

async def detect_object():
    try:
        global picam2, interpreter, labels
        camera_was_running = hasattr(picam2, '_running') and picam2._running
        if not camera_was_running:
            picam2.start()
            await asyncio.sleep(0.5)
        frame = picam2.capture_array("main")
        if not camera_was_running:
            picam2.stop()
        if frame is None:
            await voice_poll.speak("Failed to capture image")
            system_status.update_detected_object("None")
            return None
        loop = asyncio.get_event_loop()
        detected_objects = await loop.run_in_executor(None, process_image, frame)
        if detected_objects:
            await voice_poll.speak(f"I see {', '.join(set(detected_objects))}")
            system_status.update_detected_object(", ".join(set(detected_objects)))
        else:
            await voice_poll.speak("Unable to identify the object")
            system_status.update_detected_object("None")
    except Exception as e:
        print(f"Object detection error: {e}")
        await voice_poll.speak("Object detection failed")
        system_status.update_detected_object("Error")

def process_image(frame):
    import cv2
    import numpy as np
    import datetime
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h, w = input_shape[1], input_shape[2]
    image_resized = cv2.resize(frame, (w, h))
    if len(input_shape) == 4 and input_shape[-1] == 1:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        image_resized = np.expand_dims(image_resized, axis=-1)
    input_data = np.expand_dims(image_resized, axis=0)
    if input_dtype == np.float32:
        input_data = np.float32(input_data) / 255.0
    else:
        input_data = input_data.astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    detected_objects = []
    im_height, im_width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(max(1, xmin * im_width))
            xmax = int(min(im_width, xmax * im_width))
            ymin = int(max(1, ymin * im_height))
            ymax = int(min(im_height, ymax * im_height))
            label_index = int(classes[i]) + 1
            label = labels[label_index] if 0 <= label_index < len(labels) else "object"
            detected_objects.append(label)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {scores[i]*100:.1f}%",
                        (xmin, max(ymin - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    now = datetime.datetime.now()
    timestamp = now.strftime("%m-%d-%y_%H-%M-%S")
    object_name = "_".join(set(detected_objects)) if detected_objects else "unknown"
    filename = f"{object_name}_{timestamp}.jpg"
    image_path = os.path.join(IMAGE_DIR, filename)
    cv2.imwrite(image_path, frame)
    return detected_objects

DEBOUNCE_MS = 250
last_press_time = {23: 0, 24: 0, 25: 0}
button1_click_count = 0
button1_first_click_time = 0
button2_click_count = 0
button2_first_click_time = 0
button3_click_count = 0
button3_first_click_time = 0

BUTTON1_HOLD_MS = 1200  # Hold threshold in ms
BUTTON2_HOLD_MS = 1200  # Hold threshold for vibration toggle
BUTTON3_HOLD_MS = 1200  # Hold threshold for distance voice toggle

async def button_monitor_loop():
    global button1_click_count, button1_first_click_time
    global button2_click_count, button2_first_click_time
    global button3_click_count, button3_first_click_time
    prev_states = {23: False, 24: False, 25: False}
    while True:
        now = time.time() * 1000
        current_states = {
            23: GPIO.input(BUTTON1_PIN) == GPIO.LOW,
            24: GPIO.input(BUTTON2_PIN) == GPIO.LOW,
            25: GPIO.input(BUTTON3_PIN) == GPIO.LOW
        }
        for pin, pressed in current_states.items():
            system_status.update_button_state(pin, pressed)
            if pressed and not prev_states[pin] and (now - last_press_time[pin]) > DEBOUNCE_MS:
                last_press_time[pin] = now
                print(f"\nBUTTON {pin} TRIGGERED!")
                if pin == 23:
                    asyncio.create_task(handle_button1_action(now))
                elif pin == 24:
                    asyncio.create_task(handle_button2_action(now))
                elif pin == 25:
                    asyncio.create_task(handle_button3_action(now))
        # Button 1 hold: repeat last received guardian message 
        if GPIO.input(BUTTON1_PIN) == GPIO.LOW:
            hold_start = now
            while GPIO.input(BUTTON1_PIN) == GPIO.LOW:
                await asyncio.sleep(0.01)
                now2 = time.time() * 1000
                if now2 - hold_start > BUTTON1_HOLD_MS:
                    last_msg = await system_status.get_last_stored_message()
                    if last_msg:
                        await voice_poll.speak("Reading the last message.", priority=1, allow_if_locked=True)
                        await voice_poll.speak(last_msg, priority=1, allow_if_locked=True)
                    else:
                        await voice_poll.speak("No inbox message.", priority=1, allow_if_locked=True)
                    while GPIO.input(BUTTON1_PIN) == GPIO.LOW:
                        await asyncio.sleep(0.01)
                    break
        
        # Button 2 hold: toggle vibration motor
        if GPIO.input(BUTTON2_PIN) == GPIO.LOW:
            hold_start = now
            while GPIO.input(BUTTON2_PIN) == GPIO.LOW:
                await asyncio.sleep(0.01)
                now2 = time.time() * 1000
                if now2 - hold_start > BUTTON2_HOLD_MS:
                    global vibration_enabled
                    vibration_enabled = not vibration_enabled
                    status = "enabled" if vibration_enabled else "disabled"
                    await voice_poll.speak(f"Vibration {status}", priority=1, allow_if_locked=True)
                    print(f"Vibration motor {status}")
                    while GPIO.input(BUTTON2_PIN) == GPIO.LOW:
                        await asyncio.sleep(0.01)
                    break
        
        # Button 3 hold: toggle distance voice-over
        if GPIO.input(BUTTON3_PIN) == GPIO.LOW:
            hold_start = now
            while GPIO.input(BUTTON3_PIN) == GPIO.LOW:
                await asyncio.sleep(0.01)
                now2 = time.time() * 1000
                if now2 - hold_start > BUTTON3_HOLD_MS:
                    global distance_voice_enabled
                    distance_voice_enabled = not distance_voice_enabled
                    status = "enabled" if distance_voice_enabled else "disabled"
                    await voice_poll.speak(f"Distance voice {status}", priority=1, allow_if_locked=True)
                    print(f"Distance voice-over {status}")
                    while GPIO.input(BUTTON3_PIN) == GPIO.LOW:
                        await asyncio.sleep(0.01)
                    break
        prev_states = current_states.copy()
        if button1_click_count > 0 and (now - button1_first_click_time) > 10000:
            button1_click_count = 0
            button1_first_click_time = 0
        if button2_click_count > 0 and (now - button2_first_click_time) > 10000:
            button2_click_count = 0
            button2_first_click_time = 0
        if button3_click_count > 0 and (now - button3_first_click_time) > 10000:
            button3_click_count = 0
            button3_first_click_time = 0
        await asyncio.sleep(0.003)

async def handle_button1_action(now):
    global button1_click_count, button1_first_click_time
    if now - button1_first_click_time > 10000:
        button1_click_count = 0
    button1_click_count += 1
    if button1_click_count == 1:
        await voice_poll.speak("Button 1 clicked. Click again within 10 seconds to confirm sending alert and location", priority=1, allow_if_locked=False)
        button1_first_click_time = now
    elif button1_click_count == 2:
        if command_lock.locked():
            await voice_poll.speak("System busy, please wait.", priority=1, allow_if_locked=True)
        else:
            await voice_poll.speak("Sending alert and location now", priority=1, allow_if_locked=True)
            lat, lon = latest_gps.lat, latest_gps.lon
            asyncio.create_task(send_sms_with_location(PHONE_NUMBER, lat, lon))
        button1_click_count = 0
        button1_first_click_time = 0

async def handle_button2_action(now):
    global button2_click_count, button2_first_click_time
    if now - button2_first_click_time > 10000:
        button2_click_count = 0
    button2_click_count += 1
    if button2_click_count == 1:
        await voice_poll.speak("Button 2 clicked. Click again within 10 seconds to confirm call", priority=1, allow_if_locked=False)
        button2_first_click_time = now
    elif button2_click_count == 2:
        if command_lock.locked():
            await voice_poll.speak("System busy, please wait.", priority=1, allow_if_locked=True)
        else:
            await voice_poll.speak("Calling now", priority=1, allow_if_locked=True)
            asyncio.create_task(make_call_async(PHONE_NUMBER))
        button2_click_count = 0
        button2_first_click_time = 0

async def handle_button3_action(now):
    global button3_click_count, button3_first_click_time
    if now - button3_first_click_time > 10000:
        button3_click_count = 0
    button3_click_count += 1
    if button3_click_count == 1:
        await voice_poll.speak("Button 3 clicked. Click again within 10 seconds to confirm object detection.", priority=1, allow_if_locked=False)
        button3_first_click_time = now
    elif button3_click_count == 2:
        if command_lock.locked():
            await voice_poll.speak("System busy, please wait.", priority=1, allow_if_locked=True)
        else:
            await voice_poll.speak("Performing object detection...", priority=1, allow_if_locked=True)
            asyncio.create_task(detect_object_non_blocking())
        button3_click_count = 0
        button3_first_click_time = 0

async def status_display_loop():
    while True:
        now = time.time()
        if now - system_status.last_display_update > 1.0:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, display_status)
            system_status.last_display_update = now
        await asyncio.sleep(0.3)

async def distance_display_loop():
    last_vibrate_time = 0
    last_distance_pattern = None
    consecutive_errors = 0
    while True:
        try:
            distance_mm = await safe_get_distance()
            if distance_mm > 0:
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    print("WARNING: Distance sensor issues detected, attempting to reinitialize...")
                    success = await safe_reinitialize_distance_sensor()
                    if success:
                        consecutive_errors = 0
                    else:
                        consecutive_errors = 3
        except Exception as e:
            print(f"Distance sensor error: {e}")
            consecutive_errors += 1
            distance_mm = -1
        if distance_mm > 0:
            distance_cm = distance_mm / 10.0
            system_status.update_distance(distance_cm)
            now = time.time()
            pattern = None
            if 30 <= distance_cm < 100:
                pattern = "one"
            elif 100 <= distance_cm < 200:
                pattern = "two"
            if (pattern != last_distance_pattern or (now - last_vibrate_time) > 3):
                if pattern == "one":
                    asyncio.create_task(vibrate_once_async(800))
                    if distance_voice_enabled:
                        await voice_poll.speak("Object detected within one meter", priority=10, allow_if_locked=False)
                elif pattern == "two":
                    asyncio.create_task(vibrate_twice_async(200, 150))
                    if distance_voice_enabled:
                        await voice_poll.speak("Object detected within two meters", priority=10, allow_if_locked=False)
                last_distance_pattern = pattern
                last_vibrate_time = now
        else:
            if consecutive_errors >= 3:
                system_status.update_distance(0)
                last_distance_pattern = None
        await asyncio.sleep(0.5)

async def send_startup_sms():
    # Initialize the modem first
    await initialize_modem()
    
    guide_text = (
        "System is online!\n"
        "Press 1 to identify the location.\n"
        "Send messages using: *your message*\n"
        "Example: *hello how are you*\n"
    )
    await send_sms_async(PHONE_NUMBER, guide_text)
    
    # Initialize message storage system with tagged system message
    await store_message("[SYSTEM] System started - message storage initialized")
    print("Message storage system initialized")

async def sms_command_listener():
    while True:
        await asyncio.sleep(3)  # Increased delay to reduce command frequency
        
        async with command_lock:  # Use the command lock to prevent interference
            # Clear buffers before checking for messages
            sim800l.reset_input_buffer()
            sim800l.reset_output_buffer()
            
            # Check for unread messages
            sim800l.write(b'AT+CMGL="REC UNREAD"\r')
            reply = await read_modem_reply_async(5)
            
            if reply and reply.strip():
                print(f"Raw SMS reply: {reply}")
                
                # Look for SMS notifications and message content
                lines = reply.split('\n')
                message_found = False
                
                # First, look for +CMGL entries which contain message metadata
                for i, line in enumerate(lines):
                    line = line.strip()
                    
                    # Found a message header with +CMGL
                    if line.startswith('+CMGL:'):
                        # The actual message content should be on the next non-empty line
                        for j in range(i + 1, len(lines)):
                            content_line = lines[j].strip()
                            if content_line and not content_line.startswith('+') and not content_line.startswith('AT') and content_line != 'OK':
                                # Found the actual message content
                                await process_sms_content(content_line)
                                message_found = True
                                break
                        if message_found:
                            break
                
                # If no +CMGL format found, try direct content processing (fallback)
                if not message_found:
                    for line in lines:
                        line = line.strip()
                        
                        # Skip system responses and metadata
                        if (not line or 
                            line.startswith('AT') or 
                            line == 'OK' or 
                            line.startswith('+CMGL:') or
                            line.startswith('+CMGR:') or
                            line.startswith('+CMTI:') or  # SMS notification
                            line.startswith('ERROR') or
                            line.startswith('>')): 
                            continue
                        
                        # Process potential message content
                        if len(line) > 1:
                            await process_sms_content(line)
                            break
            
            # Clean up processed messages
            sim800l.write(b'AT+CMGDA="DEL READ"\r')
            await read_modem_reply_async(2)
            
            # Clear buffers after operations
            sim800l.reset_input_buffer()

async def call_monitor_listener():
    """Monitor for incoming calls and auto-block them"""
    while True:
        await asyncio.sleep(0.5)  # Check more frequently for faster blocking
        
        # Check if there's any incoming data from the modem
        if sim800l.in_waiting > 0:
            try:
                # Read without using the command lock to avoid blocking SMS operations
                data = sim800l.read(sim800l.in_waiting).decode(errors='ignore')
                
                if data:
                    print(f"Modem notification: {data.strip()}")
                    
                    # Check for incoming call notifications - block immediately
                    if '+CLIP:' in data or 'RING' in data:
                        print("Incoming call detected - auto blocking")
                        await voice_poll.speak("Call blocked", priority=5, allow_if_locked=True)
                        asyncio.create_task(block_call_async())
                        
                    # Check for other call status notifications
                    elif '+CRC:' in data and 'VOICE' in data:
                        print("Voice call status detected - blocking")
                        asyncio.create_task(block_call_async())
                    
                    # Also block on any call-related unsolicited responses
                    elif 'NO CARRIER' in data or 'BUSY' in data:
                        print("Call terminated/blocked successfully")
                        
            except Exception as e:
                print(f"Error monitoring calls: {e}")
                await asyncio.sleep(1)

async def process_sms_content(content):
    """Process the actual SMS content"""
    content = content.strip()
    
    print(f"Processing SMS content: {content}")
    
    # Location request command - check this FIRST before ANY filtering
    # Handle both exact "1" and stripped content that becomes "1"
    if content.strip() == '1' or (len(content.strip()) == 1 and content.strip() == '1'):
        print("Location request received - sending GPS coordinates")
        # Notify user that guardian is requesting location
        await voice_poll.speak("Guardian is requesting your location", priority=1, allow_if_locked=True)
        lat, lon = latest_gps.lat, latest_gps.lon
        asyncio.create_task(send_sms_with_location(PHONE_NUMBER, lat, lon))
        return
    
    # Enhanced filtering for AT commands and system messages
    at_commands = [
        "AT+CMGL", "AT+CMGDA", "AT+CMGS", "AT+CMGF", "ATD", "ATE0", "AT+CPMS",
        "REC UNREAD", "DEL READ", "FC2", "+CMGL:", "+CMGS:", "+CMTI:", "+CMGR:",
        "System is online", "Message sent", "OK", "ERROR", ">"
    ]
    
    # Enhanced filtering for system/technical messages
    def is_system_message(msg):
        msg_upper = msg.upper()
        
        # Check if it's an AT command or response
        if any(cmd in msg_upper for cmd in at_commands):
            return True
            
        # Filter very short messages that are likely system codes (but NOT "1" since we already handled it)
        if len(msg) <= 3 and msg != '1':
            return True
            
        # Filter messages that are just numbers or codes (but NOT "1")
        if msg.isdigit() and len(msg) <= 4 and msg != '1':
            return True
            
        # Filter messages with only uppercase letters and numbers (likely system codes)
        if msg.isupper() and len(msg) <= 10 and not msg.isalpha():
            return True
            
        # Filter common system patterns
        system_patterns = ['FC', 'AT', 'CMG', 'SIM', 'GSM', 'GPRS', 'SMS']
        if any(pattern in msg_upper for pattern in system_patterns) and len(msg) <= 15:
            return True
            
        return False
    
    # Skip if empty, system message, or AT command
    if not content or len(content) <= 1 or is_system_message(content):
        print(f"Filtered out system/AT command: {content}")
        return
    
    # STRICT: Only accept guardian messages with exact *message* format
    if content.startswith('*') and content.endswith('*') and len(content) > 2:
        # Extract the message between the asterisks
        user_msg = content[1:-1].strip()  # Remove starting and ending "*"
        if user_msg and len(user_msg) > 1:  # Ensure meaningful message
            print(f"Received guardian message: {user_msg}")
            system_status.update_received_sms(content)
            await store_message(f"[GUARDIAN] {user_msg}")
            
            # Just notify about new message, don't read it automatically
            await voice_poll.speak("You received a new message", priority=1, allow_if_locked=True)
            
            # Clean up old messages periodically
            await clear_old_messages(10)
            return
    
    # REJECT all other messages that don't match the strict *message* format
    else:
        print(f"Rejected message (invalid format): {content}")
        print("Only messages in format *your message* are accepted")

async def main_loop():
    print("\nEnhanced Distance Detection + Multi-Button Alert System Running...")
    print("=" * 50)
    while True:
        await asyncio.sleep(0.01)

def cleanup_resources():
    try:
        if 'tof' in globals():
            tof.stop_ranging()
            tof.close()
    except Exception as e:
        print(f"Distance sensor cleanup error: {e}")
    try:
        GPIO.cleanup()
    except:
        pass
    try:
        if 'pi' in globals() and pi.connected:
            pi.bb_serial_read_close(GPS_RX)
    except Exception as e:
        print(f"GPS cleanup error: {e}")
    try:
        if 'sim800l' in globals():
            sim800l.close()
    except Exception as e:
        print(f"SIM800L cleanup error: {e}")
    try:
        if 'picam2' in globals() and picam2:
            picam2.stop()
            picam2.close()
    except Exception as e:
        print(f"Camera cleanup error: {e}")

async def run_all():
    await send_startup_sms()
    asyncio.create_task(voice_poll.voice_over_loop())
    asyncio.create_task(sms_command_listener())
    asyncio.create_task(call_monitor_listener())
    try:
        await asyncio.gather(
            gps_background_loop(),
            distance_display_loop(),
            button_monitor_loop(),
            status_display_loop(),
            main_loop(),
        )
    finally:
        cleanup_resources()

try:
    asyncio.run(run_all())
except KeyboardInterrupt:
    print("STOPPING...")
    cleanup_resources()
except Exception as e:
    print(f"ERROR: {e}")
    cleanup_resources()
