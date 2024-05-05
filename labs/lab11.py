import numpy as np

from classes.in_out import In_Out
from classes.analysis import Analysis
from classes.processing import Processing
from classes.model import Model

in_out = In_Out()
processing = Processing()
model = Model()
analysis = Analysis()

is_color = False


def plot_threshold(img_data, limit, name):
    threshold = processing.threshold(img_data, limit)

    in_out.show_jpg_files([img_data, threshold],
                          [name, 'threshold'],
                          is_color)

    return threshold


def main():
    model_name = 'model'
    grace_name = 'grace'

    m = 64
    dt = 1
    sigma = 0.8
    mask = 3

    # sigma - стандартное отклонение Гауссовой функции, определяющее степень размытия

    def plot(img_name, fc):
        # Оригинальное изображение
        img_data = in_out.read_jpg_file(img_name + '/' + img_name)

        # Зашумленное изображение
        random_noise = model.noise_2d(img_data.shape, 0.01)
        noisy_data = img_data + random_noise

        # Отфильтрованное изображение
        filtered_data = processing.average_filter(noisy_data, mask)
        filtered_data = np.array(filtered_data)

        in_out.show_jpg_files([img_data, noisy_data, filtered_data],
                              ['original', 'noised image',
                               f'average filter (mask {mask}x{mask})'],
                              is_color)

        def without_noise():
            threshold = plot_threshold(img_data, 200, 'original')

            lpf_image = processing.lpf_2d(threshold, sigma)
            hpf_image = processing.hpf_2d(threshold, fc, dt, m)

            in_out.show_jpg_files([threshold,
                                   processing.threshold(lpf_image, 50),
                                   processing.threshold(hpf_image, 50)],
                                  ['threshold', 'LPF', 'HPF'],
                                  is_color)

        def with_noise():
            threshold = plot_threshold(noisy_data, 200, 'noised image')

            lpf_image = processing.lpf_2d(threshold, sigma)
            hpf_image = processing.hpf_2d(threshold, fc, dt, m)

            in_out.show_jpg_files([threshold,
                                   processing.threshold(lpf_image, 50),
                                   processing.threshold(hpf_image, 50)],
                                  ['threshold', 'LPF', 'HPF'],
                                  is_color)

        def filtered():
            threshold = plot_threshold(filtered_data, 200, f'average filter (mask {mask}x{mask})')

            lpf_image = processing.lpf_2d(threshold, sigma)
            hpf_image = processing.hpf_2d(threshold, fc, dt, m)

            in_out.show_jpg_files([threshold,
                                   processing.threshold(lpf_image, 50),
                                   processing.threshold(hpf_image, 50)],
                                  [f'threshold', 'LPF', 'HPF'],
                                  is_color)

        without_noise()
        with_noise()
        filtered()

    plot(model_name, 0.07)
    plot(grace_name, 0.05)
