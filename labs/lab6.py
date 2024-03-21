import matplotlib.pyplot as plt
import numpy as np
import sys

from classes.in_out import In_Out
from classes.processing import Processing
from classes.analysis import Analysis


# Экземпляры классов
in_out = In_Out()
processing = Processing()
analysis = Analysis()

if_color = False
np.set_printoptions(threshold=sys.maxsize)


# Параметры
height = 256
dt = 1
m = 16
indent = 0.05

# Для циклов
start = 0
stop = 70
step = 10


def main():
    # Полноэкранный режим
    plt.rcParams["figure.figsize"] = [20, 8.5]
    plt.rcParams["figure.autolayout"] = True

    img_name = 'c12-85v'
    # img_name = 'u0'

    img_data = in_out.read_jpg_file(img_name + '/' + img_name)

    # data_xn = new_analysis.spectrum_fourier([_ for _ in range(img_data.shape[1])], img_data.shape[1], dt)
    # data_xn_cut = new_analysis.spectrum_fourier([_ for _ in range(img_data.shape[1] - 1)], img_data.shape[1] - 1, dt)
    #
    # # Детектор артефактов
    # diff = count_auto_correlation_and_print(img_data, data_xn, data_xn_cut)
    # max_freq = count_cross_correlation_and_print(diff, data_xn_cut)

    max_freq = 0.2933  # Для c12-85v
    # max_freq = 0.3882  # Для u0

    # Подавитель артефактов
    filtered_img = change(max_freq, img_data)

    in_out.write_jpg_file(filtered_img, img_name + '/' + img_name + '_without_artifacts')


def plot(original, changed):
    plt.figure()
    plt.subplot(121)
    in_out.show_jpg_sub(original, if_color, 'original')
    plt.subplot(122)
    in_out.show_jpg_sub(changed, if_color, 'changed')
    plt.show()


def count_auto_correlation_and_print(data, xn, xn_cut):
    fig1, ax1 = plt.subplots(nrows=(stop - start) // step, ncols=3)

    diff = []
    
    for i in range(start, stop, step):
        row = data[i]
        row_fourier = analysis.fourier(row)  # Спектр Фурье строки

        diff.append(processing.anti_trend_linear(row))
        diff_fourier = analysis.fourier(diff[-1])  # Спектр Фурье производной строки

        acf = analysis.auto_correlation(diff[-1])
        acf_fourier = analysis.fourier(acf)  # Спектр Фурье acf производной строки

        m = i // step

        def plot_fourier(array, x_n, name_array, n):
            x_values = np.linspace(0, 0.5, len(array) // 2 - 1)
            y_values = array[1:len(array) // 2]

            ax1[m, n].plot(x_values, y_values)

            if m == 0:
                ax1[m, n].set_title(name_array, fontsize=15)

            if n == 0:
                ax1[m, n].set_ylabel('row_' + str(i), fontsize=15)

            max_point = max(array[1:len(array) // 2])
            index = array[1:len(array) // 2].index(max_point)
            freq = x_n[index]

            if freq != 0.0:
                ax1[m, n].annotate('freq: ' + str(round(freq, 4)), xy=(freq, max_point), xytext=(freq + 0.02, max_point / 2))
                
        plot_fourier(row_fourier, xn, 'row_spectrum', 0)
        plot_fourier(diff_fourier, xn_cut, 'differential_spectrum', 1)
        plot_fourier(acf_fourier, xn_cut, 'acf_spectrum', 2)
        
    plt.show()
    return diff


def count_cross_correlation_and_print(diff, xn_cut):
    max_freq = 0
    cols = 3
    rows = ((stop - start) // step) // cols
    count = 0

    fig2, ax2 = plt.subplots(nrows=rows, ncols=cols)

    for i in range(rows):
        for j in range(cols):
            ccf = analysis.cross_correlation(diff[count], diff[count + 1])
            ccf_fourier = analysis.fourier(ccf)   # Спектр Фурье ccf производных строк

            x_values = np.linspace(0, 0.5, len(ccf_fourier) // 2 - 1)
            y_values = ccf_fourier[1:len(ccf_fourier) // 2]

            ax2[i, j].plot(x_values, y_values)
            ax2[i, j].set_title('row_' + str(count * step) + ' and row_' + str((count + 1) * step), fontsize=15)

            max_point = max(ccf_fourier[0:len(ccf_fourier) // 2])
            index = ccf_fourier[0:len(ccf_fourier) // 2].index(max_point)
            freq = xn_cut[index]

            if freq > max_freq:
                max_freq = freq

            ax2[i, j].annotate('freq: ' + str(round(freq, 4)), xy=(freq, max_point), xytext=(freq + 0.02, max_point / 2))
            ax2[i, j].vlines([freq - indent, freq + indent], 0, max(ccf_fourier), color='gray', linestyle='dashed')

            count += 1

    plt.show()
    return max_freq


def change(max_freq, img_data):
    fc1 = max_freq - indent
    fc2 = max_freq + indent

    bsw = processing.bsf(fc1, fc2, dt, m)

    filtered = np.empty((0, img_data.shape[1]), dtype=int)

    for i in range(img_data.shape[0]):
        row = img_data[i, :]
        new_row = analysis.convolution(row, bsw, len(row), 2 * m + 1)
        new_row_int = np.array(new_row, int)

        filtered = np.insert(filtered, filtered.shape[0], new_row_int, axis=0)

    filtered = np.delete(filtered, slice(m), 1)

    # reshape
    coef = height / img_data.shape[0]
    img_small = in_out.reshape_bilinear_interpolation(img_data, coef)
    filtered_small = in_out.reshape_bilinear_interpolation(filtered, coef)

    plot(img_small, filtered_small)
    return filtered
