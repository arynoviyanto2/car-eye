import numpy as np


def get_intersection_area(rect1, rect2):
    r1_x1 = rect1[0]
    r1_x2 = rect1[0] + rect1[2]

    r2_x1 = rect2[0]
    r2_x2 = rect2[0] + rect2[2]

    w = np.min([r1_x2, r2_x2]) - np.max([r1_x1, r2_x1])
    w = 0 if w < 0 else w

    r1_y1 = rect1[1]
    r1_y2 = rect1[1] + rect1[3]

    r2_y1 = rect2[1]
    r2_y2 = rect2[1] + rect2[3]

    h = np.min([r1_y2, r2_y2]) - np.max([r1_y1, r2_y1])
    h = 0 if h < 0 else h

    return w * h


def assign_cars(cars, current_cars, debug=False):
    delta_grad = 0.1
    delta_movement = 800
    delta_tick = 15

    for car in cars:
        car['active'] = False

    if len(cars) != 0:
        for candidate in current_cars:
            new_car_flag = True
            for car in reversed(cars):
                delta_x = candidate['center'][0] - car['center'][0] + np.finfo(float).eps
                delta_y = candidate['center'][1] - car['center'][1] + np.finfo(float).eps

                # gradient = 1.0 * delta_y / delta_x
                # movement = delta_x * delta_x + delta_y * delta_y
                tick_grad = candidate['tick'] - car['tick']

                # print("tick_grad: {0}, gradient = {1:.2f}, movement = {2}".format(tick_grad, gradient, movement))
                # if tick_grad <= delta_tick and movement <= delta_movement * tick_grad:

                inter_area = get_intersection_area(candidate['rect'], car['rect'])
                # print(inter_area / (car['rect'][2] * car['rect'][3]))
                if inter_area > (0.7 - tick_grad * 0.02) * car['rect'][2] * car['rect'][3] and tick_grad <= delta_tick:
                    # print(inter_area)
                    if debug:
                        prev_car = {
                            'id': car['id'],
                            'center': car['center'],
                            'rect': car['rect'],
                            'tick': car['tick'],
                            'active': True
                        }
                        cars.append(prev_car)

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
