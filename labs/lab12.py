import matplotlib.pyplot as plt
import numpy as np

from classes.in_out import In_Out
from classes.processing import Processing
from classes.model import Model


def filter_with_mask(data, mask):
    out_data = np.zeros(data.shape)

    for row in range(1, data.shape[0] - 1):
        for col in range(1, data.shape[1] - 1):
            # Подматрица изображения размерем 3х3
            part = data[row - 1: row + 2, col - 1: col + 2]

            # Свертка подматрицы и маски
            new_el = np.sum(mask * part)

            if new_el < 0:
                new_el = 0
            if new_el > 255:
                new_el = 255

            out_data[row, col] = new_el

    return out_data


def filter_with_gradient(data, mask1, mask2):
    out_data = np.zeros(data.shape)

    for row in range(1, data.shape[0] - 1):
        for col in range(1, data.shape[1] - 1):
            # Подматрица изображения размерем 3х3
            part = data[row - 1: row + 2, col - 1: col + 2]

            # Свертка подматрицы и маски 1
            new_el1 = np.sum(mask1 * part)
            # Свертка подматрицы и маски 2
            new_el2 = np.sum(mask2 * part)

            # Величина градиента = Евклидово расстояние (по теореме Пифагора)
            new_el = np.sqrt(new_el1 ** 2 + new_el2 ** 2)

            if new_el < 0:
                new_el = 0
            if new_el > 255:
                new_el = 255

            out_data[row, col] = new_el

    return out_data


mask_laplacian = [np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]]),
                  np.array([[0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]]),
                  np.array([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]]),
                  np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])]

mask_prewitt = [np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]]),
                np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]]),
                np.array([[0, 1, 1],
                          [-1, 0, 1],
                          [-1, -1, 0]]),
                np.array([[-1, -1, 0],
                          [-1, 0, 1],
                          [0, 1, 1]])]

mask_sobel = [np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]),
              np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]),
              np.array([[0, 1, 2],
                        [-1, 0, 1],
                        [-2, -1, 0]]),
              np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 2]])]

one_laplacian = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])


def main():
    # Экземпляры классов
    in_out = In_Out()
    processing = Processing()
    model = Model()

    # Некоторые значения
    model_name = 'model'
    grace_name = 'grace'
    birches_name = 'birches'

    is_color = False

    def task1(img_name, mask=3):
        # Оригинальное изображение
        img_data = in_out.read_jpg_file(img_name + '/' + img_name)

        # Зашумленное изображение
        random_noise = model.noise_2d(img_data.shape, 0.01)
        noisy_data = img_data + random_noise

        # Отфильтрованное изображение
        filtered_data = processing.average_filter(noisy_data, mask)
        filtered_data = np.array(filtered_data)

        mask1, mask2 = 1, 2

        in_out.show_jpg_files([img_data, noisy_data, filtered_data],
                              ['original', 'noised image',
                               f'average filter (mask {mask}x{mask})'],
                              is_color)

        def without_noise():
            count = 1
            rows, cols = 3, 4

            # Laplacian
            for i in range(len(mask_laplacian)):
                plt.subplot(rows, cols, count)
                in_out.show_jpg_sub(filter_with_mask(img_data, mask_laplacian[i]), is_color,
                                    f'Laplacian mask {i + 1}')
                count += 1

            # Prewitt
            for i in range(len(mask_prewitt)):
                plt.subplot(rows, cols, count)
                in_out.show_jpg_sub(filter_with_mask(img_data, mask_prewitt[i]), is_color,
                                    f'Prewitt mask {i + 1}')
                count += 1

            # Sobel
            for i in range(len(mask_sobel)):
                plt.subplot(rows, cols, count)
                in_out.show_jpg_sub(filter_with_mask(img_data, mask_sobel[i]), is_color,
                                    f'Sobel mask {i + 1}')
                count += 1

            plt.show()

        def with_noise():
            # Laplacian
            laplacian = filter_with_mask(noisy_data, mask_laplacian[2])
            laplacian_threshold = processing.threshold(laplacian, 50)

            in_out.show_jpg_files([laplacian, laplacian_threshold],
                                  ['Laplacian mask 3', 'threshold'],
                                  is_color)

            # Prewitt
            prewitt = filter_with_gradient(noisy_data, mask_prewitt[mask1 - 1], mask_prewitt[mask2 - 1])
            prewitt_threshold = processing.threshold(prewitt, 50)

            in_out.show_jpg_files([prewitt, prewitt_threshold],
                                  [f'Prewitt mask {mask1} and {mask2}', 'threshold'],
                                  is_color)

            # Sobel
            sobel = filter_with_gradient(noisy_data, mask_sobel[mask1 - 1], mask_sobel[mask2 - 1])
            sobel_threshold = processing.threshold(sobel, 50)

            in_out.show_jpg_files([sobel, sobel_threshold],
                                  [f'Sobel mask {mask1} and {mask2}', 'threshold'],
                                  is_color)

        def filtered():
            # Laplacian
            laplacian = filter_with_mask(filtered_data, mask_laplacian[2])
            laplacian_threshold = processing.threshold(laplacian, 15)

            in_out.show_jpg_files([laplacian, laplacian_threshold],
                                  ['Laplacian mask 3', 'threshold'],
                                  is_color)

            # Prewitt
            prewitt = filter_with_gradient(filtered_data, mask_prewitt[mask1 - 1], mask_prewitt[mask2 - 1])
            prewitt_threshold = processing.threshold(prewitt, 50)

            in_out.show_jpg_files([prewitt, prewitt_threshold],
                                  [f'Prewitt mask {mask1} and {mask2}', 'threshold'],
                                  is_color)

            # Sobel
            sobel = filter_with_gradient(filtered_data, mask_sobel[mask1 - 1], mask_sobel[mask2 - 1])
            sobel_threshold = processing.threshold(sobel, 50)

            in_out.show_jpg_files([sobel, sobel_threshold],
                                  [f'Sobel mask {mask1} and {mask2}', 'threshold'],
                                  is_color)

        without_noise()
        with_noise()
        filtered()

    # Выделение контуров birches однопроходным Лапласианом
    def task2():
        img_data = in_out.read_jpg_file(birches_name + '/' + birches_name)

        laplacian = filter_with_mask(img_data, one_laplacian)

        in_out.write_jpg_file(laplacian, birches_name + '/' + birches_name + '_changed')

        in_out.show_jpg_files([img_data, laplacian],
                              ['original', f'Laplacian'],
                              is_color)

    task1(model_name)
    task1(grace_name)
    task2()
