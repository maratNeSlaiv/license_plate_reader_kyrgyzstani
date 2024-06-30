class NotGettingFrame(Exception):
    def __init__(self) -> None:
        super().__init__('Frame is None')