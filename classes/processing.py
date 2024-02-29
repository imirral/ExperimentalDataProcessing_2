import math
import numpy as np

class Processing:
    def anti_shift(self, inData, N):
        out_data = inData
        c = sum(inData) / len(inData)
        for i in range(len(inData)):
            out_data[i] = inData[i] - c
        return out_data

    def anti_spike(self, data, N, R):
        out_data = []
        for i in range(N):
            if (data[i] > R or data[i] < -R) and i != 0 and i != N - 1:
                out_data.append((data[i - 1] + data[i + 1]) / 2)
            else:
                out_data.append(data[i])
        return out_data

    def anti_trend_linear(self, data, N):
        out_data = []
        for i in range(N - 1):
            out_data.append(data[i + 1] - data[i])
        return out_data

    def anti_trend_nonlinear(self, data, N, W):
        out_data = []
        for i in range(N - W):
            x_n = 0
            for k in range(W):
                x_n += data[i + k]
            x_n = x_n / W
            out_data.append(x_n)
        return out_data

    def anti_noise(self, data, N, M):
        out_data = []
        print(len(data))
        for i in range(N):
            element = 0
            for j in range(M):
                # print(i, j)
                element += data[j][i]
            element = element / M
            out_data.append(element)
        return out_data

    def lpf(self, fc, dt, m):
        d = [0.35577019, 0.2436983, 0.07211497, 0.00630165]

        fact = 2 * fc * dt
        lpw = []
        lpw.append(fact)
        arg = fact * math.pi
        for i in range(1, m + 1):
            lpw.append(np.sin(arg * i) / (math.pi * i))

        lpw[m] = lpw[m] / 2

        sumg = lpw[0]
        for i in range(1, m + 1):
            sum = d[0]
            arg = math.pi * i / m
            for k in range(1, 4):
                sum += 2 * d[k] * np.cos(arg * k)
            lpw[i] = lpw[i] * sum
            sumg += 2 * lpw[i]
        for i in range(m + 1):
            lpw[i] = lpw[i] / sumg
        return lpw

    def reflect_lpf(self, lpw):
        reflection = []
        for i in range(len(lpw) - 1, 0, -1):
            reflection.append(lpw[i])
        reflection.extend(lpw)
        return reflection

    def hpf(self, fc, dt, m):
        lpw = self.reflect_lpf(self.lpf(fc, dt, m))
        hpw = []
        Loper = 2 * m + 1
        for k in range(Loper):
            if k == m:
                hpw.append(1 - lpw[k])
            else:
                hpw.append(- lpw[k])
        return hpw

    def bpf(self, fc1, fc2, dt, m):
        lpw1 = self.reflect_lpf(self.lpf(fc1, dt, m))
        lpw2 = self.reflect_lpf(self.lpf(fc2, dt, m))
        bpw = []
        Loper = 2 * m + 1
        for k in range(Loper):
            bpw.append(lpw2[k] - lpw1[k])
        return bpw

    def bsf(self, fc1, fc2, dt, m):
        lpw1 = self.reflect_lpf(self.lpf(fc1, dt, m))
        lpw2 = self.reflect_lpf(self.lpf(fc2, dt, m))
        bsw = []
        Loper = 2 * m + 1
        for k in range(Loper):
            if k == m:
                bsw.append(1. + lpw1[k] - lpw2[k])
            else:
                bsw.append(lpw1[k] - lpw2[k])
        return bsw

    def recount(self, data, R):
        x_min = min(data)
        x_max = max(data)

        for i in range(len(data)):
            data[i] = ((data[i] - x_min) / (x_max - x_min) - 0.5) * 2 * R
        return data

    def recount_2d(self, array, s):
        new_arr = array.copy()
        min_val = np.min(new_arr)
        max_val = np.max(new_arr)

        if max_val - min_val == 0:
            print("Division by zero or near-zero denominator!")

        # Масштабирование для снижения вероятности переполнения
        scale_factor = 1.0
        if (max_val - min_val) > 0:
            scale_factor = s / (max_val - min_val)

        for i in range(new_arr.shape[0]):
            for j in range(new_arr.shape[1]):
                new_arr[i, j] = (new_arr[i, j] - min_val) * scale_factor

        return new_arr

    def negative(self, image, max_intensity):
        neg_image = max_intensity - image
        return neg_image

    def gamma_transform(self, image, c_gamma, gamma):
        gamma_corrected = np.power(c_gamma * image / 255.0, gamma) * 255.0
        gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

        return gamma_corrected

    def log_transform(self, image, c_log):
        eps = 1e-8  # Минимальное значение

        log_transformed = c_log * np.log(eps + image)
        log_transformed = np.clip(log_transformed, 0, 255).astype(np.uint8)

        return log_transformed