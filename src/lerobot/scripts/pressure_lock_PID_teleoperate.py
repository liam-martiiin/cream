import time
import serial
import threading
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Update these imports to match your project's directory structure
from lerobot.teleoperators.so_leader.config_so_leader import SOSimplifiedLeaderPIDConfig
from lerobot.robots.so_follower.config_so_follower import SOSimplifiedFollowerPIDConfig
from lerobot.teleoperators.so_leader.so_simplified_leader_PID import SOSimplifiedLeaderPID
from lerobot.robots.so_follower.so_simplified_follower_PID import SOSimplifiedFollowerPID

stop_event = threading.Event()  # Event to signal threads to stop

# define gamma (absolute angle where marker tip is vertical)
gamma = 1.776

# --- Hardware Configuration ---
ARDUINO_PORT = "/dev/ttyACM1" # Replace with your Arduino Nano Every port
BAUD_RATE = 115200

# --- PID Tuning Values ---
Kp = 0.00055
Ki = 0.00
Kd = 0.00

# --- Motor Safety Limits ---
# LeRobot is outputting degrees. We use +/- 100 to leave a small 
# 2-degree safety buffer to protect the physical joint.
MIN_WRIST_POS = -100.0 
MAX_WRIST_POS = 100.0

# --- Global State Variables ---
current_pressure = 0.0
target_pressure = 0.0
pressure_locked = False
locked_z_pos = 0.0

# --- PID State ---
integral = 0.0
prev_error = 0.0
prev_time = time.perf_counter()

# --- GUI State Variables ---
# Using standard lists to retain all historical data points
time_history = []
pressure_history = []
gui_start_time = time.perf_counter()

def read_arduino():
    """Background thread to poll the latest load cell pressure via serial."""
    global current_pressure
    try:
        # Added a short timeout so readline() doesn't hang indefinitely
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
        
        # Give the Arduino 2 seconds to reboot after opening the serial connection
        time.sleep(2)
        ser.reset_input_buffer() # Clear any startup garbage from the buffer
        
        while True:
            # Request a reading from the Arduino
            ser.write(b'P')
            
            # Read the response
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    current_pressure = float(line)
                except ValueError:
                    continue
            
            # Small delay to prevent overwhelming the serial bus
            time.sleep(0.01) 
            
    except Exception as e:
        print(f"Warning: Could not connect to Arduino at {ARDUINO_PORT}: {e}")

def listen_for_input():
    """Background thread to listen for 'p' + ENTER to toggle the PID lock."""
    global pressure_locked, target_pressure, current_pressure, integral, prev_error, prev_time
    while True:
        # sys.stdin.readline waits until ENTER is pressed
        user_input = sys.stdin.readline().strip()
        if user_input.lower() == 'p':
            pressure_locked = not pressure_locked
            if pressure_locked:
                target_pressure = current_pressure
                
                # Reset PID state to prevent integral windup from past movements
                integral = 0.0
                prev_error = 0.0
                prev_time = time.perf_counter()
                
                # Added extra newlines to prevent overwriting the live pressure feed
                print(f"\n\n[LOCKED] Target Pressure Set To: {target_pressure}")
            else:
                print("\n\n[UNLOCKED] Control returned to leader arm.")

def teleoperation_worker():
    """Background thread to run the high-frequency control loop."""
    global locked_z_pos, integral, prev_error, prev_time

    # 1. Start Background Threads
    threading.Thread(target=read_arduino, daemon=True).start()
    threading.Thread(target=listen_for_input, daemon=True).start()

    # 2. Initialize Hardware
    print("Connecting hardware...")
    
    # Configure LEADER (Using Lab Fork ID Registry)
    lead_config = SOSimplifiedLeaderPIDConfig()
    lead_config.id = "right_leader"  # Make sure you registered the leader with this name!
    lead_config.cameras = {}
    leader = SOSimplifiedLeaderPID(lead_config)
    
    # Configure FOLLOWER (Using Lab Fork ID Registry)
    follow_config = SOSimplifiedFollowerPIDConfig()
    follow_config.id = "right_follower"
    follow_config.cameras = {}
    follower = SOSimplifiedFollowerPID(follow_config)
    
    follower.connect()
    leader.connect()
    
    # Print throttle variables
    last_print_time = time.perf_counter()
    
    # 3. Main Teleoperation Loop
    try:
        while not stop_event.is_set():
            
            # --- Record Data for GUI ---
            time_history.append(time.perf_counter() - gui_start_time)
            pressure_history.append(current_pressure)

            # 1. Read follower state
            follower_obs = follower.get_observation_radians()
            t1 = follower_obs["shoulder_pan.pos"]
            t2 = follower_obs["shoulder_lift.pos"]
            t3 = follower_obs["elbow_flex.pos"]
            t4 = follower_obs["wrist_flex.pos"]
            
            # Grab the requested action from the leader arm
            action = leader.get_action()
        
            if pressure_locked:
                
                current_time = time.perf_counter()
                dt = current_time - prev_time
                
                if dt > 0:
                    # Calculate Error
                    error = target_pressure - current_pressure
                    
                    # Calculate PID terms
                    proportional = Kp * error
                    integral += (error * dt)
                    derivative = (error - prev_error) / dt
                    
                    output = proportional + (Ki * integral) - (Kd * derivative)
                    if t4 < gamma - t3 - t2: # applies PID in correct direction based on wrist angle
                    # Apply the PID output offset to the locked joint position
                        locked_z_pos += output
                    else: # applies PID in correct direction based on wrist angle
                        locked_z_pos -= output
                    
                    # --- NEW: CLAMP THE POSITION ---
                    # This prevents integer overflows and stops the arm from breaking itself
                    locked_z_pos = max(MIN_WRIST_POS, min(MAX_WRIST_POS, locked_z_pos))
                    
                    # Override the leader's command for the Z-axis
                    action['wrist_flex.pos'] = locked_z_pos
                    
                    # Store state for next iteration
                    prev_error = error
                    prev_time = current_time
            else:
                # When unlocked, passively track the leader's wrist position. 
                # This ensures that when the lock is engaged, the PID starts exactly 
                # where the user physically left the arm.
                locked_z_pos = action.get('wrist_flex.pos', 0.0)

            # Dispatch the final, combined action to the follower
            follower.send_action(action)
            
            # --- Live Console Print ---
            current_time_loop = time.perf_counter()
            if current_time_loop - last_print_time >= 0.1: # Update console at 10Hz
                status_text = f"LOCKED @ {target_pressure:.1f}" if pressure_locked else "UNLOCKED"
                # Using \r to overwrite the line, padded with spaces to erase old artifacts
                sys.stdout.write(f"\rLive Pressure: {current_pressure:>6.1f} | Status: {status_text:<20}")
                sys.stdout.flush()
                last_print_time = current_time_loop
            
            # Match the loop rate to the Feetech serial bus capabilities (~100Hz)
            time.sleep(0.01)
            
    finally:
        print("\n\nDisconnecting...")
        follower.disconnect()
        leader.disconnect()


def update_plot(frame, line, ax):
    """Callback function for FuncAnimation to redraw the graph."""
    if len(time_history) > 0:
        line.set_data(time_history, pressure_history)
        
        # Expand the x-axis to fit all data from start to current time
        current_time = time_history[-1]
        ax.set_xlim(0, max(10, current_time + 1))
    return line,

def main():
    # 1. Start the arm control loop as a background daemon thread
    teleop_thread = threading.Thread(target=teleoperation_worker)
    teleop_thread.start()

    # 2. Set up the Matplotlib GUI on the main thread
    fig, ax = plt.subplots()
    ax.set_ylim(-500, 500)
    ax.set_title("Live Load Cell Pressure")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure")
    ax.grid(True)
    
    line, = ax.plot([], [], lw=2, color='blue')
    
   # Update the plot every 100ms (10Hz) to prevent GUI freezing
    ani = FuncAnimation(fig, update_plot, fargs=(line, ax), interval=100, blit=False, cache_frame_data=False)
    
    try:
        # plt.show() blocks the main thread, keeping the application alive
        plt.show()
    except KeyboardInterrupt:
        pass # Catch the Ctrl+C quietly
    finally:
        print("\nClosing GUI and signaling arm to shut down...")
        stop_event.set()       # Tell the while loop to stop
        teleop_thread.join()   # Wait for the arm to finish disconnecting

if __name__ == "__main__":
    main()