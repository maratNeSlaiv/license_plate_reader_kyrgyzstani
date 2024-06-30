from typing import Optional
from uuid import UUID

from pydantic import BaseModel, IPvAnyAddress


class MotionDetected(BaseModel):
    motion_detected: bool
    pk: str
    ip_address: str
    type_barier: str
    username: str
    password: str


class Number(BaseModel):
    number_text: str
    country: str


class RecordingSuccessResponse(BaseModel):
    status_code: int
    message: str
    data: Number


class CarInFromCamera(BaseModel):
    number_region: str
    car_number: str
    car_color: Optional[str] = None
    car_model: Optional[str] = None
    photo_in: Optional[str] = None


class CarOutFromCamera(BaseModel):
    number_region: str
    car_number: str
    car_color: Optional[str] = None
    car_model: Optional[str] = None
    photo_out: Optional[str] = None


class ResponseFromBackSchema(BaseModel):
    status_type: str
    status_code: int
    message: Optional[str]