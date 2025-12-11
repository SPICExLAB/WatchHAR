# SAMoSA Dataset Constants and Mappings

# Activity class mappings for SAMoSA dataset (27 classes, alphabetically ordered)
# Matches audioIMU reference implementation exactly
SAMOSA_CLASS_LABEL_MAPPING = {
    "Alarm_clock": 0,
    "Blender_in_use": 1,
    "Brushing_hair": 2,
    "Chopping": 3,
    "Clapping": 4,
    "Coughing": 5,
    "Drill in use": 6,
    "Drinking": 7,
    "Grating": 8,
    "Hair_dryer_in_use": 9,
    "Hammering": 10,
    "Knocking": 11,
    "Laughing": 12,
    "Microwave": 13,
    "Pouring_pitcher": 14,
    "Sanding": 15,
    "Scratching": 16,
    "Screwing": 17,
    "Shaver_in_use": 18,
    "Toilet_flushing": 19,
    "Toothbrushing": 20,
    "Twisting_jar": 21,
    "Vacuum in use": 22,
    "Washing_Utensils": 23,
    "Washing_hands": 24,
    "Wiping_with_rag": 25,
    "Other": 26
}

# Context groupings for SAMoSA activities (matches audioIMU)
SAMOSA_CONTEXTS = {
    'Bathroom': [
        'Brushing_hair', 'Hair_dryer_in_use', 'Shaver_in_use',
        'Toilet_flushing', 'Toothbrushing', 'Washing_hands'
    ],
    'Kitchen': [
        'Blender_in_use', 'Chopping', 'Grating', 'Microwave',
        'Pouring_pitcher', 'Twisting_jar', 'Washing_Utensils', 'Wiping_with_rag'
    ],
    'Misc': [
        'Alarm_clock', 'Clapping', 'Coughing', 'Drinking',
        'Knocking', 'Laughing', 'Scratching'
    ],
    'Workshop': [
        'Drill in use', 'Hammering', 'Sanding', 'Screwing', 'Vacuum in use'
    ]
}

# Reverse mapping from class index to activity name
SAMOSA_INDEX_TO_ACTIVITY = {v: k for k, v in SAMOSA_CLASS_LABEL_MAPPING.items()}

# Context mapping from activity to context
ACTIVITY_TO_CONTEXT = {}
for context, activities in SAMOSA_CONTEXTS.items():
    for activity in activities:
        ACTIVITY_TO_CONTEXT[activity] = context

# Sensor configurations
SENSOR_INDICES = {
    'acc': slice(0, 3),      # Accelerometer data (x, y, z)
    'gyro': slice(3, 6),     # Gyroscope data (x, y, z)
    'rotvec': slice(6, 9)    # Rotation vector data (x, y, z)
}

# Default data parameters
DEFAULT_IMU_SR = 50          # IMU sampling rate (Hz)
DEFAULT_AUDIO_SR = 16000     # Original audio sampling rate (Hz) - raw data
DEFAULT_AUDIO_SR_TARGET = 1000  # Target audio sampling rate (Hz) - 1kHz for edge_har_chi25
DEFAULT_AUDIO_SR_SPEECH = 1000  # Audio sampling rate for speech (Hz)
DEFAULT_IMU_WIN_SEC = 1      # IMU window length in seconds
DEFAULT_HOP_LENGTH = 10      # Hop length for windowing (IMU samples) - edge_har_chi25

# STFT parameters
STFT_WINDOW_LENGTH_SECONDS = 0.025  # 25ms window
STFT_HOP_LENGTH_SECONDS = 0.01      # 10ms hop -> produces 96 frames
AUDIO_EXAMPLE_WINDOW_SECONDS = 0.96  # 0.96 second audio window -> 96 frames

# Model feature dimensions
MODEL_FEATURE_DIMS = {
    'imu': {
        'cnn1d': 256,
        'cnn2d': 128,
        'convlstm': 128,
        'attend_discriminate': 128
    },
    'audio': {
        'dymn04': 512,
        'dymn10': 1280,
        'dymn20': 2560,
        'mn05': 640
    }
}

# Available model types
AVAILABLE_IMU_MODELS = ['cnn2d']  # Only CNN2D is implemented
AVAILABLE_AUDIO_MODELS = ['mn05']  # Only MN05 is supported
AVAILABLE_FUSION_MODELS = ['individualgf']  # Only IndividualGatedFusion is implemented

# Data file patterns
DATA_FILE_PATTERN = r'(\d+)---(.+)---(.+)---(\d+)\.pkl'  # ParticipantID---Context---Activity---TrialNo.pkl