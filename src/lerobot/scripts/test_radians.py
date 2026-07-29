#!/usr/bin/env python
import os
import sys
import time

# --- AUTOMATIC PATH PATCH ---
# This looks 2 levels up from 'src/lerobot/scripts/' to find the 'src/' folder
# and adds it to Python's search list so it can find the 'lerobot' package.
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(src_dir)

# --- CORRECTED IMPORTS FOR YOUR LAYOUT ---
from lerobot.robots.so_follower.so_simplified_follower_PID import SOSimplifiedFollowerPID
from lerobot.robots.so_follower.config_so_follower import SOSimplifiedFollowerPIDConfig

def main():
    print("Initializing configuration...")
    # 1. Set up the hardware configuration object
    config = SOSimplifiedFollowerPIDConfig()
    
    # ASSIGN THE FRIENDLY NAME INSTEAD OF A PORT
    # This triggers your fork's resolve_or_verify_port logic
    config.id = "right_follower" 
    
    # Disable cameras for drawing to maximize your loop speed
    config.cameras = {} 
    
    # 2. Instantiate your custom robot class
    print("Creating robot instance...")
    arm = SOSimplifiedFollowerPID(config)
    
    # 3. Connect to the arm (this loads your json calibration offsets)
    print("Connecting to the arm... (Press ENTER when prompted to use your calibration file)")
    arm.connect()
    arm.bus.disable_torque()
    
    print("\n--- Streaming Raw Radians ---")
    print("Move the arm joints by hand. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # 4. Call your custom radian observation method
            angles = arm.get_observation_radians()
            
            # Format the output nicely for the terminal
            print(
                f"Pan: {angles['shoulder_pan.pos']:6.3f} rad | "
                f"Lift: {angles['shoulder_lift.pos']:6.3f} rad | "
                f"Elbow: {angles['elbow_flex.pos']:6.3f} rad | "
                f"Wrist: {angles['wrist_flex.pos']:6.3f} rad", 
                end="\r" # Overwrites the current line dynamically
            )
            
            time.sleep(0.05) # Read at ~20Hz for testing
            
    except KeyboardInterrupt:
        print("\n\nStopping stream...")
    finally:
        # 5. Always disconnect to safely release the serial port
        arm.disconnect()
        print("Disconnected safely.")

if __name__ == "__main__":
    main()
    
