import math
import numpy as np
import random
import time


class Model:

    def trend_linear(self, N, a, b):
        t = np.arange(0, N, 1)
        x = a * t + b
        return x

    def trend_nonlinear(self, N, a, b):
        t = np.arange(0, N, 1)
        x = b * np.exp(a * t)
        return x

    def noise(self, N, R):
        data = []
        for i in range(N):
            xk = random.random()
            data.append(xk)
        recount_data = self.recount(data, R)
        return recount_data

    def noise_2d(self, image_shape):
        rows, cols = image_shape
        noise_data = []

        for _ in range(rows):
            noise_row = self.noise(cols, R=255)
            noise_data.append(noise_row)

        return noise_data

    def impulse_noise_2d(self, image_data):
        rows = len(image_data)
        cols = len(image_data[0])

        result = [[0 for _ in range(cols)] for _ in range(rows)]

        m = random.randint(int(cols * 0.005), int(cols * 0.01))

        for i in range(rows):
            impulse_noise_row = self.impulse_noise(image_data[i], cols, m, R=255, Rs=25)
            result[i] = impulse_noise_row

        return result

    def my_noise(self, N, R):
        m = 32768
        a = 23
        c = 12345
        seed = time.time()
        data = [0 for _ in range(N)]
        data[0] = seed
        for i in range(1, N):
            data[i] = math.fmod((a * data[i - 1] + c), m)
        for i in range(N):
            data[i] = data[i] / 10 ** len(str(int(data[i])))
        recount_data = self.recount(data, R)
        return recount_data

    def shift(self, inData, C):
        outData = inData
        for i in range(len(inData)):
            outData[i] = inData[i] + C
        return outData

    def shift_2d(self, array, c):
        c_arr = np.full(array.shape, c)
        return array + c_arr

    def mult_model_2d(self, array, c):
        c_arr = np.full(array.shape, c)
        return array * c_arr

    def impulse_noise(self, data, N, M, R, Rs):
        out_data = data.copy()
        d = dict()
        for i in range(M):
            key = random.randint(0, N - 1)
            rand = random.random() * 2 * Rs + (R - Rs)
            sign = random.choice([-1, 1])
            d[key] = sign * rand
        for key in d.keys():
            out_data[key] = d[key]
        return out_data

    def harm(self, N, A, f, del_t):
        k = np.arange(0, N, 1)
        x = A * np.sin(2 * math.pi * f * del_t * k)
        return x

    def poly_harm(self, N, A0, f0, A1, f1, A2, f2, del_t):
        k = np.arange(0, N, 1)
        xi0 = A0 * np.sin(2 * math.pi * f0 * del_t * k)
        xi1 = A1 * np.sin(2 * math.pi * f1 * del_t * k)
        xi2 = A2 * np.sin(2 * math.pi * f2 * del_t * k)
        x = xi0 + xi1 + xi2
        return x

    def add_model(self, data1, data2, N):
        outData = []
        for i in range(N):
            outData.append(data1[i] + data2[i])
        return outData

    def mult_model(self, data1, data2, N):
        outData = []
        for i in range(N):
            outData.append(data1[i] * data2[i])
        return outData

    def heart(self, N, f, del_t, a):
        harm = self.harm(N, 1, f, del_t)
        nonlinear_trend = self.trend_nonlinear(N, -a * del_t, 1)
        h_t = self.mult_model(harm, nonlinear_trend, N)
        h_n = []
        max_ht = max(h_t)
        for i in range(len(h_t)):
            h_n.append(h_t[i] * 120 / max_ht)
        return h_n

    def rhythm(self, N, M, R, Rs):
        x_t = [0 for _ in range(N)]
        for i in range(N):
            if i % M == 0 and i != 0:
                x_t[i] = random.random() * 2 * Rs + (R - Rs)
        return x_t

    def pw(self, c1, c2, n1, n2, n3, n4, N):
        pw = [1 for _ in range(n1)]
        pw.extend([c1 for _ in range(n1, n2 + 1)])
        pw.extend([1 for _ in range(n2 + 1, n3)])
        pw.extend([c2 for _ in range(n3, n4 + 1)])
        pw.extend([1 for _ in range(n4 + 1, N)])
        return pw

    def recount(self, data, R):
        x_min = min(data)
        x_max = max(data)

        for i in range(len(data)):
            data[i] = ((data[i] - x_min) / (x_max - x_min) - 0.5) * 2 * R

        return data

    # s = ((r - min) / (max - min)) * S
    def recount_2d(self, array, s=255):
        new_arr = array.copy()

        min = np.min(new_arr)
        max = np.max(new_arr)

        if max - min == 0:
            print("Знаменатель равен 0")

        # Масштабирование для снижения вероятности переполнения
        if (max - min) > 0:
            scale_factor = s / (max - min)
        else:
            scale_factor = 1.0

        # Приведение в шкалу серости
        for i in range(new_arr.shape[0]):
            for j in range(new_arr.shape[1]):
                new_arr[i, j] = (new_arr[i, j] - min) * scale_factor

        return new_arr

    def get_rectangle(self, background_size, rect_size):
        M, N = background_size
        m, n = rect_size

        image = np.zeros((M, N))  # Создание черного изображения фона

        # Поиск координат для центрирования прямоугольника
        start_row = M // 2 - m // 2
        end_row = start_row + m
        start_col = N // 2 - n // 2
        end_col = start_col + n

        # Заполнение белого прямоугольника в центре черного фона
        image[start_row:end_row, start_col:end_col] = 255

        return image