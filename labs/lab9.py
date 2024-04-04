import matplotlib.pyplot as plt
import numpy as np

from classes.model import Model
from classes.analysis import Analysis
from classes.processing import Processing
from classes.in_out import In_Out


def plot_graph(item, data, name, fourier=False, xn=None, dt=0.002):
    if xn is None:
        xn = []

    plt.subplot(item)
    plt.title(name)

    if fourier:
        plt.plot(xn, data)
        plt.xlim([0, 1 / (dt * 2)])
    else:
        plt.plot(data)


def main():
    model = Model()
    analysis = Analysis()
    processing = Processing()
    in_out = In_Out()

    if_color = False

    def task1():
        n = 10 ** 3
        m = 200
        a = 30
        f = 7
        r = 1
        rs = 0.1
        del_t = 0.005

        h = model.heart(n, f, del_t, a)
        x = model.rhythm(n, m, r, rs)
        y = analysis.convolution(x, h, n, m)

        def a():
            y_fourier = analysis.fourier_sep(y)  # Комплексн. спектр кардиограммы
            h_fourier = analysis.fourier_sep(h)  # Комплексн. спектр ф-ции сердечн. мышцы

            x_spectre = processing.complex_division(y_fourier, h_fourier)
            x_spectre_inverse = analysis.inverse_fourier(x_spectre)

            plt.figure()
            plot_graph(411, h, 'h(t)')
            plot_graph(412, x, 'x(t)')
            plot_graph(413, y, 'y(t)')
            plot_graph(414, x_spectre_inverse, "restored x(t)")
            plt.show()

        def b(a=0.01):
            noise = model.noise(n, max(y) / 100)
            y_noised = model.add_model(y, noise, n)

            y_fourier = analysis.fourier_sep(y_noised)  # Комплексн. спектр зашумленн. кардиограммы
            h_fourier = analysis.fourier_sep(h)   # Комплексн. спектр ф-ции сердечн. мышцы

            x_spectre = processing.complex_noised_division(y_fourier, h_fourier, a)
            x_spectre_inverse = analysis.inverse_fourier(x_spectre)

            plt.figure()
            plot_graph(411, h, 'h(t)')
            plot_graph(412, x, 'x(t)')
            plot_graph(413, y_noised, 'noised y(t)')
            plot_graph(414, x_spectre_inverse, f"restored x(t) (a = {a})")
            plt.show()

        a()
        b(0.05)

    def task2():
        #  Искажающая функция
        h_file_name = 'kern64L.dat'
        h = in_out.read_dat_file(h_file_name)

        shape = (185, 259)
        for i in range(h.size, shape[1]):
            h = np.append(h, 0)

        h_fourier = analysis.fourier_sep(h)  # Комплексн. спектр искажающей ф-ции

        plt.plot(h)
        plt.suptitle('kern64L.dat')
        plt.show()

        def subtask_a():
            #  Смазанное изображение
            shape = (185, 259)
            file_name = 'blur259x185L.dat'
            file_data = in_out.read_dat_file(file_name)
            img_data = np.asarray(file_data).reshape(shape)

            # Фильтрация
            filtered = np.empty((0, img_data.shape[1]), dtype="float32")

            for i in range(img_data.shape[0]):
                row = img_data[i, :]

                g_fourier = analysis.fourier_sep(row)  # Комплексн. спектр строки изображен.

                x_spectre = processing.complex_division(g_fourier, h_fourier)
                x_spectre_inverse = analysis.inverse_fourier(x_spectre)

                filtered = np.insert(filtered, filtered.shape[0], np.asarray(x_spectre_inverse), axis=0)

            plt.figure()
            plt.subplot(121)
            in_out.show_jpg_sub(img_data, if_color, 'blur259x185L.dat')
            plt.subplot(122)
            in_out.show_jpg_sub(filtered, if_color, 'filtered blur259x185L.dat')
            plt.show()

        def subtask_b(a=0.01):
            #  Смазанное изображение
            shape = (185, 259)
            file_name = 'blur259x185L_N.dat'
            file_data = in_out.read_dat_file(file_name)
            img_data = np.asarray(file_data).reshape(shape)

            # Фильтрация
            filtered = np.empty((0, img_data.shape[1]), dtype="float32")

            for i in range(img_data.shape[0]):
                row = img_data[i, :]

                g_fourier = analysis.fourier_sep(row)   # Комплексн. спектр строки изображен.

                x_spectre = processing.complex_noised_division(g_fourier, h_fourier, a)
                x_spectre_inverse = analysis.inverse_fourier(x_spectre)

                filtered = np.insert(filtered, filtered.shape[0], np.asarray(x_spectre_inverse), axis=0)

            plt.figure()
            plt.subplot(121)
            in_out.show_jpg_sub(img_data, if_color, 'blur259x185L_N.dat')
            plt.subplot(122)
            in_out.show_jpg_sub(filtered, if_color, f'filtered blur259x185L_N.dat (a = {a})')
            plt.show()

        subtask_a()
        subtask_b(0.1)

    task1()
    task2()
