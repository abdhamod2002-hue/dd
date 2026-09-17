CONF_THRES = 0.25               # Lowered from 0.5 to capture low-contrast/small detections
IOU_THRES = 0.7
TRACK_ACTIVATION_THRESHOLD = 0.25 # Lowered from 0.5 for earlier track activation
MINIMUM_MATCHING_THRESHOLD = 0.35 # Lowered from 0.5 for second-pass low-conf matching
LOST_TRACK_BUFFER = 30
MINIMUM_CONSECUTIVE_FRAMES = 2    # Prevent ID skipping (set to 2)

PERSON_CLASS_ID = 0
TRASH_BAG_CLASS_ID = 1
