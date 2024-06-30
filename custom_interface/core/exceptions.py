class CameraConnectionError(Exception):
    def __init__(self, message: object) -> None:
        super().__init__('Not connectet to ip camera {}'.format(message))