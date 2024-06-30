import psycopg2

from custom_interface.core.settinngs import settings
from custom_interface.core.schemas import Camera

class DataBase:

    def __init__(self) -> None:
        self.con = psycopg2.connect(settings.PG_CONF)
        self.cur = self.con.cursor()
    
    def get_all_cameras(self):
        self.cur.execute('SELECT ip_camera, camera_username, camera_password FROM barrier_point where is_active=true;')
        return self.cur.fetchall()
    
    def get_cameras(self) -> list[Camera]:
        instanses = self.get_all_cameras()
        cameras: list[Camera | None] = []
        for camera in instanses:
            ip_address, username, password = camera
            cameras.append(Camera(
                ip_address=ip_address,
                username=username,
                password=password,
            ))
        return cameras


    def __del__(self):
        if hasattr(self, 'cur'):
            self.cur.close()
        if hasattr(self, 'con'):
            self.con.close()