# Project Overview:
Indoor localization systems are quite expensive. The project is trying to address this problem by provide an affordable solution for local educational institutions. It project allows researchers to conduct various control experiments that heavily depends on localization, one of which is controlling a swarm of drones.

<p align="center">
  <img src="https://github.com/UbaydaAbdulsamad/DroneMocap_App/assets/80641961/417fca31-d630-4881-93bb-0313ebf4808f">
</p>

# Results
The system was able to locate an object **4 meters** away from the camera with precision of **±1cm @60 fps**. The following video represents two conducted tests for the accuracy of measurements (more on them later).

https://github.com/UbaydaAbdulsamad/DroneMocap_App/assets/80641961/8ad20179-4d1e-4d45-b2c5-09588646b20a

# A use case for the project
The System has been used to track a single drone. The following sections will demonstrate this use case along with the working principle of the system as whole.

**Note**:
Active markers in this project have not been integrated yet (which would solve the problem of motion blur and latency resulted from the use of commercial components). It'll be added as soon as I have time to finish this project.

## System Architecture
The motion capture system is composed of two cameras, two Raspberry Pi modules and a PC. To leverage active markers, an infrared sensor is used with each drone unite.

<p align="center">
  <img src="https://github.com/UbaydaAbdulsamad/DroneMocap_App/assets/80641961/eca27583-d0f3-4661-8d17-5066c60fa640">
</p>

## Working principle
It's important to limit the data flow in order to achieve better performance. On board markers detection (on Raspberry Pi) is done to achieve that since small sized arrays of data are sent over Gigabit Ethernet to the PC. An IR LED on one of the Raspberry Pi modules sends periodic light pulses to be recieved by all the drones' active markers. A pulse will trigger the active markers on all drones and in return emit light pulse right when a frame is being captured.

<p align="center">
  <img src="https://github.com/UbaydaAbdulsamad/DroneMocap_App/assets/80641961/d0f4b719-abed-4e80-9050-04bf039607ec">
</p>

## Dealing with motion blur
To capture fast moving objects without motion blur, one should consider using high shutter speed cameras, but they're quite expensive. A work around for this problem -which has been tested and worked extremely well- is to illuminate the object (the IR LEDs) for a short amount of time while the frame is being captured.

<p align="center">
  <img src="https://github.com/UbaydaAbdulsamad/DroneMocap_App/assets/80641961/25fb3fd3-40bb-4dfb-99ce-98eb9821d428" height=250>
</p>

## Sensor Fusion and Latency
To ensure supreme performance with affordable components, the latency our system will have due to image processing and triangulation could be calculated and eliminated completely using a timer on the MCU. Kalman filter will fuse the two measurement sources (IMU & our system) into an estimate of the location which will solve the drifting problem IMUs have.
<p align="center">
  <img src="https://github.com/UbaydaAbdulsamad/DroneMocap_App/assets/80641961/6f6efca5-bc63-477d-9d71-26ddcd3ae241" height=250>
</p>

# Tests & Results
To test the system we first calibrated the cameras setup using chessboard with known dimensions then conducted the following tests: 
- **One meter bar test**:<br>
  Two LEDs fixed on each tip of a 1 meter bar. The bar is swung around in 3D space and the system estimated two 3D locations that found by triangulation to be on meter apart.

<p align="center">
  <img src="https://github.com/UbaydaAbdulsamad/DroneMocap_App/assets/80641961/a5eb1151-b9cd-4b5e-90e9-1adb11c6d116" height=200>
</p>

- **White board test**:<br>
  Multiple real measurements were compared to the estimates of the system. The distance between the two 3D locations was found to have an error of ±1cm.

