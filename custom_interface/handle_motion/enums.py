from enum import Enum


class CreateCarStatuses(Enum):

    car_check_created = 100
    the_car_in_parking = 101
    the_blockpost_not_availible = 103
    barrier_not_found = 105
    car_id_not_found = 104
    the_blockpost_busy = 106
    car_success_out = 107
    car_has_debt = 108
    car_check_activated = 109
    car_check_deactivated = 110