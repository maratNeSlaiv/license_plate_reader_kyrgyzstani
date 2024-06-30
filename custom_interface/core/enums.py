from enum import Enum

from http import HTTPStatus

class StatusType(Enum):
    SUCCESS = 'success'
    ERROR = 'error'


class ResponseStatuses(Enum):

    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message

    SUCCESS_SAVED_IMAGE = (HTTPStatus.CREATED, 'Image from blockpost seccess saved')

    VALIDATION_ERROR = (HTTPStatus.UNPROCESSABLE_ENTITY, 'Validation fields error')


class BarrierStatuses(Enum):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message

    ready = (100, 'barrier ready to work')
    in_prossesing = (110, 'barrier is busy in the moment')
    is_not_active = (120, 'barrier is not active in the moment')


class RecordingStatuses(Enum):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message

    number_detected = (100, 'Detected car and her number')
    number_not_detected = (104, 'Number or car not recording')


class BarrierCurrentStatuses(Enum):

    busy = 104
    available = 100


class NumberCountryTypes(Enum):
    kg = 'kg'
    am = 'am'
    rus = 'rus'
    eu = 'eu'
    ge = 'ge'
