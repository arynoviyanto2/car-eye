import numpy as np

def assign_cars(cars, current_cars):
    delta_grad = 0.1
    delta_movement = 800
    delta_tick = 15

    new_car_flag = True

    for car in cars:
        car['active'] = False

    if len(cars) != 0:
        for candidate in current_cars:
            for car in reversed(cars):
                delta_x = candidate['center'][0] - car['center'][0] + np.finfo(float).eps
                delta_y = candidate['center'][1] - car['center'][1] + np.finfo(float).eps

                gradient = 1.0 * delta_y / delta_x
                movement = delta_x * delta_x + delta_y * delta_y
                tick_grad = candidate['tick'] - car['tick']

                print("tick_grad: {0}, gradient = {1:.2f}, movement = {2}".format(tick_grad, gradient, movement))
                if tick_grad <= delta_tick and movement <= delta_movement * tick_grad:
                    car['center'] = candidate['center']
                    car['rect'] = candidate['rect']
                    car['tick'] = candidate['tick']
                    car['active'] = True
                    new_car_flag = False
                    break

            if new_car_flag:
                candidate['id'] = len(cars) + 1
                candidate['active'] = True
                cars.append(candidate)
    else:
        for candidate in current_cars:
            candidate['id'] = len(cars) + 1
            candidate['active'] = True
            cars.append(candidate)
