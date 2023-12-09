import numpy as np

def accumulator(image):
    
    for i, row in enumerate(image[1:], start = 1):
        for j, val in enumerate(row):
            image[i, j] += np.min([image[i - 1, np.max([0, j - 1])], 
                                   image[i - 1, j], 
                                   image[i - 1, np.min([j + 1, image.shape[1] - 1])]])

    return image

def optimal_path(accumulator_image, optimum, optimum_idx):
    if optimum_idx:
        # find range of indices of interest
        min_range = np.max([0, optimum_idx[-1] - 1])
        max_range = np.min([optimum_idx[-1] + 1, accumulator_image.shape[1] - 1]) + 1
        
        # create temporary array to index in
        temp = np.full(shape = accumulator_image.shape[1], fill_value = np.inf)
        temp[min_range:max_range] = accumulator_image[-1, min_range: max_range]
        
        # update list of optimal values
        optimum_idx += [np.argmin(temp)]
    else:
        # this is the first time
        optimum_idx += [np.argmin(accumulator_image[-1])]
    
    # update values for optimal path
    optimum += [accumulator_image[-1, optimum_idx[-1]]]
    
    if len(accumulator_image) == 1:
        return optimum_idx[::-1], optimum[::-1]
    else:
        return optimal_path(accumulator_image[:-1], optimum, optimum_idx)


def path_tracing(image):
    a_image = accumulator(image)
    o_path = optimal_path(a_image, [], [])
    
    return a_image, o_path