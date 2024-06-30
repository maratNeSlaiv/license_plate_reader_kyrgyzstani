from enum import Enum

class RecordingStatuses(Enum):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message

    number_detected = (100, 'Detected car and her number')
    number_not_detected = (104, 'Number or car not recording')