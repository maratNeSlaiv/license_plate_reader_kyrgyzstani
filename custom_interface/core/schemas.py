from typing import List, Optional
from pydantic import BaseModel
from http import HTTPStatus

from custom_interface.core.enums import StatusType

class BaseResponse(BaseModel):
    status: str
    status_type: str
    message: str
    _status_code: int

    class Config:
        attribute_types_allowed = True

    @property
    def status_code(self) -> int:
        return self._status_code


class ValidationErrorResponse(BaseResponse):
    status: str = StatusType.ERROR.value
    status_type: str = HTTPStatus.UNPROCESSABLE_ENTITY.name
    message: str = HTTPStatus.UNPROCESSABLE_ENTITY.phrase
    errors: List = {
        "field_name": ["validation error message"],
        "another_field_name": ["validation error message"],
    }


class Camera(BaseModel):

    ip_address: str
    username: str
    password: str


class SleepCommandToMotionDetecter(BaseModel):
    sleep_seconds: int
    ip_address: str


class BarrierCameraStatus(BaseModel):
    barrier_pk: str
    status_code: int
