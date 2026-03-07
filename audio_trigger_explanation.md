# How the Audio Trigger Works in PySlidemorpher

## Overview
The audio trigger system in PySlidemorpher creates a dynamic slideshow that responds to music by automatically transitioning between images based on audio characteristics. Think of it as a smart DJ for your photos that "listens" to the music and changes images at the perfect moments.

## How It Works

### 1. Enhanced Audio Analysis
The system continuously analyzes the playing audio file and extracts ten key characteristics:

**Time-Domain Features:**
- **Intensity (RMS)**: The overall loudness or energy level of the audio at any moment
- **Peak Amplitude**: The highest volume spike in a short time window
- **Zero Crossing Rate**: Measure of how often the audio signal crosses zero (indicates noisiness/pitch)

**Frequency-Domain Features:**
- **Spectral Centroid**: A measure of the "brightness" or high-frequency content of the sound
- **Spectral Rolloff**: The frequency below which 85% of the energy is contained
- **Low Frequency Energy**: Energy in the bass range (20-250 Hz)
- **Mid Frequency Energy**: Energy in the midrange (250-4000 Hz)
- **High Frequency Energy**: Energy in the treble range (4000-11000 Hz)

**Advanced Beat Detection:**
- **Beat Strength**: Energy variance indicating rhythm and beats
- **Onset Strength**: Spectral flux analysis for precise beat timing detection

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

### 4. Enhanced Audio-Responsive Transitions
When a trigger fires, the transition itself intelligently adapts to the audio characteristics:

**Speed Modulation:**
- **Intensity Factor**: Louder audio creates faster transitions (0.5x to 2.0x normal speed)
- **Beat Factor**: Strong beats further accelerate transitions (0.8x to 1.5x multiplier)
- **Combined Speed**: Both factors work together for dynamic speed control

**Visual Detail Adaptation:**
- **Pixel Size**: Higher intensity creates smaller pixels for more detailed transitions
- **Range**: Adaptive pixel size ranges from 50% to 100% of base setting

**Frequency-Based Effects:**
- **High Frequency Content**: Sharp, crisp sounds trigger sharper easing curves (cubic)
- **Bass-Heavy Content**: Low frequencies trigger smoother easing curves (sine)
- **Balanced Content**: Uses the user-selected easing function

**Transition Selection:**
- **Random Mode**: Intelligently selects from all available transition types
- **Specific Effects**: Swarm, tornado, swirl, drip, rainfall, and more

### 5. Subtle Effects Between Transitions
Even when not transitioning, the system adds subtle audio-reactive effects:
- **Brightness Modulation**: Images get slightly brighter during louder audio moments

### 6. Technical Implementation
The system runs multiple threads simultaneously:

- **Audio Monitor Thread**: Continuously analyzes audio features (updates every 10ms)
- **Frame Generator Thread**: Creates transition frames and manages image changes
- **Main Display Thread**: Shows the frames on screen at the correct frame rate

**Technical Specifications:**
- **Sample Rate**: 22.05 kHz for audio analysis
- **Analysis Windows**: 
  - Short window: 50ms for immediate response
  - Long window: 200ms for beat detection
  - FFT window: 100ms for frequency analysis
- **History Tracking**:
  - Audio features: Last 0.5 seconds (50 samples at 100Hz)
  - Intensity history: Last 1 second (100 samples)
  - Beat history: Last 0.2 seconds (20 samples)
- **FFT Analysis**: Hanning windowed with spectral flux calculation
- **Frequency Bands**: Bass (20-250Hz), Mids (250-4000Hz), Highs (4000-11000Hz)
- **Adaptive Timing**: 200ms to 500ms intervals based on beat strength

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
