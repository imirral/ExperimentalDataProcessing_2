import math
import numpy as np
import random


class Model:
    def trend_nonlinear(self, N, a, b):
        t = np.arange(0, N, 1)
        x = b * np.exp(a * t)
        return x

    def noise(self, n, r):
        data = []
        for i in range(n):
            xk = random.random()
            data.append(xk)
        recount_data = self.recount(data, r)
        return recount_data

    def noise_2d(self, image_shape, noise_level=0.1, r=255):
        rows, cols = image_shape
        noise_data = []

        for _ in range(rows):
            noise_row = []
            for _ in range(cols):
                if random.random() < noise_level:
                    noise_row.append(random.random() * r)
                else:
                    noise_row.append(0)
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