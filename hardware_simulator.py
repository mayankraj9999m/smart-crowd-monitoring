"""
hardware_simulator.py
Demonstration and integration module for IoT / Physical Safety Actuators.
Connects with Arduino/ESP32 via Serial or simulates physical alerts:
- Green LED: Safe Crowd Flow
- Amber LED: Moderate Bottleneck Warning
- Red Strobe Light + Piezo Siren: Overcrowding / Stampede Hazard Alert!
"""

import time
import argparse


class CrowdSafetyActuator:
    def __init__(self, port="COM3", baudrate=9600, mock=True):
        self.mock = mock
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        
        if not self.mock:
            try:
                import serial
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
                print(f"[Hardware] Connected to physical controller on {self.port} at {self.baudrate} baud.")
            except Exception as e:
                print(f"[Hardware Warning] Could not open {self.port}: {e}. Falling back to Simulation Mode.")
                self.mock = True

    def trigger_alert(self, crowd_count, status_level):
        """
        Sends control commands to microcontroller based on crowd level:
        'G': Green Safe
        'Y': Yellow Caution
        'R': Red Emergency Alarm
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if status_level == "CRITICAL":
            command = b"R\n"
            pin_status = "PIN 13 (PIEZO SIREN): HIGH | PIN 12 (RED STROBE): BLINKING (5Hz)"
            action = "🚨 SOUNDING EMERGENCY SIREN & LOCKING ENTRY TURNSTILES!"
        elif status_level == "WARNING":
            command = b"Y\n"
            pin_status = "PIN 11 (AMBER LED): SOLID ON | PIN 13: LOW"
            action = "⚠️ SLOWING AUTOMATED INGRESS GATES"
        else:
            command = b"G\n"
            pin_status = "PIN 10 (GREEN LED): SOLID ON | PIN 13: LOW"
            action = "✅ NORMAL PEDESTRIAN FLOW"
            
        print(f"\n[{timestamp}] Telemetry: Detected Count = {crowd_count:.0f} | Status = {status_level}")
        print(f" -> Hardware State: {pin_status}")
        print(f" -> Automated Safety Action: {action}")
        
        if not self.mock and self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(command)

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crowd Safety IoT Actuator Demo")
    parser.add_argument("--mock", action="store_true", default=True, help="Run in software simulation mode")
    parser.add_argument("--port", type=str, default="COM3", help="Serial COM Port for Arduino")
    args = parser.parse_args()
    
    actuator = CrowdSafetyActuator(port=args.port, mock=args.mock)
    
    print("\n--- Testing Hardware Safety Scenarios ---")
    time.sleep(1)
    actuator.trigger_alert(crowd_count=35, status_level="SAFE")
    time.sleep(2)
    actuator.trigger_alert(crowd_count=120, status_level="WARNING")
    time.sleep(2)
    actuator.trigger_alert(crowd_count=340, status_level="CRITICAL")
    time.sleep(1)
    actuator.trigger_alert(crowd_count=45, status_level="SAFE")
    actuator.close()

