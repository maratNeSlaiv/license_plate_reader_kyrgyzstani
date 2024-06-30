from typing import Any, Optional

from pydantic import BaseModel

from custom_interface.statuses import RecordingStatuses


class BaseResponse(BaseModel):
    status_code: int
    message: str


class BaseResponseSuccess(BaseModel):
    status_code: int
    message: str
    data: Optional[Any]


class BaseErrorResponse(BaseModel):
    detail: Any


class Number(BaseModel):
    number_text: str
    country: str


class RecordingSuccessResponse(BaseResponseSuccess):
    status_code: int = RecordingStatuses.number_detected.status_code
    message: str = RecordingStatuses.number_detected.message
    data: Number = Number(number_text='01234ABC', country='kg')


class RecordingNotFound(BaseResponse):
    status_code: int = RecordingStatuses.number_not_detected.status_code
    message: str = RecordingStatuses.number_not_detected.message


class RecordingNotFoundResponse(BaseErrorResponse):
    detail: RecordingNotFound = dict(RecordingNotFound())
