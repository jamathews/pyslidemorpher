# How the Audio Trigger Works in PySlidemorpher

## Overview
The audio trigger system in PySlidemorpher creates a dynamic slideshow that responds to music by automatically transitioning between images based on audio characteristics. Think of it as a smart DJ for your photos that "listens" to the music and changes images at the perfect moments.

## How It Works

### 1. Audio Analysis
The system continuously analyzes the playing audio file and extracts four key characteristics:

- **Intensity (RMS)**: The overall loudness or energy level of the audio at any moment
- **Peak Amplitude**: The highest volume spike in a short time window
- **Beat Strength**: How much the energy varies (indicating rhythm and beats)
- **Spectral Centroid**: A measure of the "brightness" or high-frequency content of the sound

### 2. Smart Trigger Detection
The system uses three different types of triggers to decide when to change images:

- **Intensity Trigger**: Fires when the audio gets louder than a dynamic threshold
- **Beat Trigger**: Activates on strong rhythmic beats (when beat strength > 0.3)
- **Peak Trigger**: Responds to sudden loud spikes in the audio

### 3. Adaptive Behavior
The system is smart and adapts to the music:

- **Dynamic Thresholds**: The trigger sensitivity adjusts based on recent audio activity - if the music has been loud, it requires even louder moments to trigger
- **Timing Control**: Stronger beats allow faster transitions (minimum 0.2 seconds between changes)
- **History Tracking**: Keeps track of the last 0.5 seconds of audio to make intelligent decisions

### 4. Audio-Responsive Transitions
When a trigger fires, the transition itself adapts to the audio:

- **Speed**: Louder audio creates faster transitions (0.5x to 2.0x normal speed)
- **Detail Level**: Higher intensity creates smaller pixel sizes for more detailed transitions
- **Transition Type**: Can use random transitions or specific effects like swarm, tornado, swirl, etc.

### 5. Subtle Effects Between Transitions
Even when not transitioning, the system adds subtle audio-reactive effects:
- **Brightness Modulation**: Images get slightly brighter during louder audio moments

### 6. Technical Implementation
The system runs multiple threads simultaneously:

- **Audio Monitor Thread**: Continuously analyzes audio features (updates every 10ms)
- **Frame Generator Thread**: Creates transition frames and manages image changes
- **Main Display Thread**: Shows the frames on screen at the correct frame rate

### 7. User Control
Users can adjust several parameters:
- **Audio Threshold**: How sensitive the system is to audio changes
- **Transition Types**: What kind of visual effects to use
- **Frame Rate**: How smooth the playback appears
- **Pixel Size**: The resolution/detail level of transitions

## The Complete Cycle
1. Music plays and the system analyzes it in real-time
2. When audio intensity, beats, or peaks exceed thresholds, a transition triggers
3. The system selects the next image and generates a transition adapted to the current audio characteristics
4. The transition plays out over a duration that matches the audio intensity
5. Between transitions, the current image may subtly pulse with the music
6. The cycle repeats, creating a dynamic, music-synchronized slideshow

This creates an immersive experience where your photos dance to the rhythm and energy of your music, with each transition perfectly timed to the audio's natural flow and intensity.