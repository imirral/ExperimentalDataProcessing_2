import numpy as np
import math

class Analysis:

    def minimum(self, data):
        return min(data)

    def maximum(self, data):
        return max(data)

    def average(self, data):
        return sum(data) / len(data)

    def dispersion(self, data):
        avg_value = self.average(data)
        disp = 0
        for i in range(len(data)):
            disp += (data[i] - avg_value) ** 2
        disp = disp / len(data)
        return disp

    def standard_deviation(self, data):
        return self.dispersion(data) ** (0.5)

    def mean_square(self, data):
        psi = 0
        for i in range(len(data)):
            psi += data[i] ** 2
        psi = psi / len(data)
        return psi

    def root_mean_square_deviation(self, data):
        return self.mean_square(data) ** (0.5)

    def asymmetry(self, data):
        avg_value = self.average(data)
        m3 = 0
        for i in range(len(data)):
            m3 += (data[i] - avg_value) ** 3
        m3 = m3 / len(data)
        return m3

    def asymmetry_coefficient(self, data):
        return self.asymmetry(data) / (self.standard_deviation(data) ** 3)

    def excess(self, data):
        avg_value = self.average(data)
        m4 = 0
        for i in range(len(data)):
            m4 += (data[i] - avg_value) ** 4
        m4 = m4 / len(data)
        return m4

    def kurtosis(self, data):
        return self.excess(data) / (self.standard_deviation(data) ** 4) - 3

    def print_statistics(self, data):
        print("1. min = ", self.minimum(data), ", max = ", self.maximum(data))
        print("2. Среднее значение: ", self.average(data))
        print("3. Дисперсия: ", self.dispersion(data))
        print("4. Стандартное отклонение: ", self.standard_deviation(data))
        print("5. Асимметрия: ", self.asymmetry(data))
        print("   Коэффициент асимметрии: ", self.asymmetry_coefficient(data))
        print("6. Эксцесс: ", self.excess(data))
        print("   Куртозис: ", self.kurtosis(data))
        print("7. Средний квадрат: ", self.mean_square(data))
        print("8. Среднеквадратическая ошибка: ", self.root_mean_square_deviation(data))

    def stationarity(self, N, data, M):
        is_stationary = True
        avg = []
        so = []
        for i in range(M):
            avg.append(self.average(data[i * N // M: (i + 1) * N // M]))
            so.append(self.standard_deviation(data[i * N // M: (i + 1) * N // M]))
        for i in range(M):
            for j in range(M):
                if i != j:
                    if abs((avg[i] - avg[j]) * 100) >= 10 or abs((so[i] - so[j]) * 100) >= 10:
                        is_stationary = False
                        break
        if is_stationary:
            print("Процесс стационарный")
        else:
            print("Процесс нестационарный")

    def hist(self, data, N, M):
        hist = dict()
        x_min = min(data)
        x_max = max(data)
        step = (x_max - x_min) / M
        for i in range(M):
            left_border = x_min + i * step
            right_border = left_border + step
            count = 0
            for j in range(N):
                if left_border <= data[j] <= right_border:
                    count += 1
            hist[left_border] = count
        return hist

    def hist_2d(self, image):
        hist = [0] * 256

        for y in range(image.shape[1]):
            for x in range(image.shape[0]):
                pixel = image[x, y]
                hist[pixel] += 1

        return hist

    def auto_correlation(self, data):
        length = len(data)

        covariance = self.covariance(data, length)
        max_r_xx = self.maximum(covariance)

        out_data = []

        for l in range(length):
            out_data.append(covariance[l] / max_r_xx)

        return out_data

    def covariance(self, data, n):
        avg_x = self.average(data)

        out_data = []

        for l in range(n):
            r_xx = 0

            for k in range(0, n - l):
                r_xx += (data[k] - avg_x) * (data[k + l] - avg_x)

            r_xx = r_xx / n
            out_data.append(r_xx)

        return out_data

    def cross_correlation(self, data_x, data_y):
        length = len(data_x)

        avg_x = self.average(data_x)
        avg_y = self.average(data_y)

        out_data = []

        for l in range(length):
            r_xy = 0

            for k in range(0, length - l):
                r_xy += (data_x[k] - avg_x) * (data_y[k + l] - avg_y)

            r_xy = r_xy / length
            out_data.append(r_xy)

        return out_data

    def convolution(self, x, h, n, m):
        out_data = []

        for i in range(n):
            y = 0

            for j in range(m):
                y += x[i - j] * h[j]

            out_data.append(y)

        return out_data

    def fourier(self, data):
        length = len(data)

        out_data = []

        for i in range(length):
            re_xn = 0
            im_xn = 0

            for k in range(length):
                re_xn += data[k] * np.cos(2 * math.pi * i * k / length)
                im_xn += data[k] * np.sin(2 * math.pi * i * k / length)

            re_xn = re_xn / length
            im_xn = im_xn / length

            xn = np.sqrt((re_xn ** 2) + (im_xn ** 2))

            out_data.append(xn)

        return out_data

    def spectrum_fourier(self, x_n, n, dt):
        out_data = []

        f_border = 1 / (2 * dt)
        df = 2 * f_border / n

        for i in range(n):
            out_data.append(x_n[i] * df)

        return out_data

    def inverse_fourier(self, data, N):
        out_data = []
        for i in range(N):
            Re_Xn = 0
            Im_Xn = 0
            for k in range(N):
                Re_Xn += data[k] * np.cos(2 * math.pi * i * k / N)
                Im_Xn += data[k] * np.sin(2 * math.pi * i * k / N)
            Re_Xn = Re_Xn / N
            Im_Xn = Im_Xn / N
            Xn = np.sqrt((Re_Xn ** 2) + (Im_Xn ** 2))
            out_data.append(Xn)
        return out_data

    def frequency_response(self, data, n):
        out_data = []

        array = self.fourier(data, n)

        for i in range(n):
            out_data.append(array[i] * n)

        return out_data
