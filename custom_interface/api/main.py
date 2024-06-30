from fastapi import FastAPI, HTTPException

from custom_interface.services import recording_service
from custom_interface.schemas import(
    RecordingSuccessResponse,
    RecordingNotFoundResponse,
    RecordingNotFound,
)

app = FastAPI()


@app.get(
    '/record_number',
    response_model=RecordingSuccessResponse,
    status_code=200,
    responses={404: {
        'model': RecordingNotFoundResponse,
        }
    }
)
def record_number(path: str):
    result = recording_service(path)
    if not result:
        raise HTTPException(status_code=404, detail=dict(RecordingNotFound()))
    return RecordingSuccessResponse(data=result)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=8060,
        reload=True,
    )