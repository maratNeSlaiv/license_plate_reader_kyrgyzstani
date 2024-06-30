from custom_interface.core.schemas import Camera
from custom_interface.consumer.consumer import split_into_subarrays, run_consumer_v2




if __name__ == '__main__':

    import torch.multiprocessing as mp
    from custom_interface.core.db import DataBase

    # db = DataBase()
    # cameras = db.get_cameras()
    # cameras = [Camera(ip_address='192.168.88.13', username='admin', password='am23hKmc'), Camera(ip_address='192.168.88.13', username='admin', password='am23hKmc')]
    cameras = [
        Camera(ip_address='192.168.88.13', username='admin', password='am23hKmc'),
        ]

    cameras_for_prosseses = split_into_subarrays(cameras)
    mp.set_start_method('spawn')
    processes = []
    for cameras_to_prosses in cameras_for_prosseses:
        print('Запускаются камеры: ', cameras_to_prosses)
        process = mp.Process(target=run_consumer_v2, args=(cameras_to_prosses,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()