import math
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import signal
from scipy.ndimage import gaussian_filter


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

    def anti_trend_linear(self, data):
        length = len(data)

        out_data = []

        for i in range(length - 1):
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

    # sigma - стандартное отклонение Гауссовой функции, определяющее степень размытия
    def lpf_2d(self, img_data, sigma):
        lpf = gaussian_filter(img_data, sigma=sigma)
        return img_data - lpf

    def hpf_2d(self, image_data, fc, dt, m):
        rows = len(image_data)
        cols = len(image_data[0])

        hpf_data = self.hpf(fc, dt, m)

        new_lines = []

        for i in range(rows):
            conv = signal.convolve(image_data[i], hpf_data, mode='same')
            new_lines.append(conv)

        new_lines = np.array(new_lines)

        erosion = np.zeros((rows, cols))

        for i in range(cols):
            conv = signal.convolve(new_lines[:, i], hpf_data, mode='same')
            erosion[:, i] = conv

        return erosion

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

    # s = L - r
    def negative(self, image, max_intensity):
        neg_image = max_intensity - image

        return neg_image

    # s = C * r ^ gamma
    def gamma_transform(self, image, c, gamma):
        gamma_corrected = np.power(c * image / 255.0, gamma) * 255.0
        gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

        return gamma_corrected

    # s = C * log(r + eps)
    def log_transform(self, image, c_log):
        eps = 1e-8  # Минимальное значение = 1 * 10 ^ (-8)

        log_transformed = c_log * np.log(eps + image)
        log_transformed = np.clip(log_transformed, 0, 255).astype(np.uint8)

        return log_transformed

    def gradation_transform(self, image, hist):
        total_pixels = image.shape[0] * image.shape[1]

        # Расчет нормализованной гистограммы (p(r_k) = n_k / M * N)
        hist_norm = [x / total_pixels for x in hist]

        # Гистограммное выравнивание (integral from 0 to r(p(q)dq))
        cdf = np.cumsum(hist_norm) * 255

        new_image = np.interp(image, range(256), cdf).astype(np.uint8)

        return new_image

    def average_filter(self, image_data, mask=3):
        rows = len(image_data)
        cols = len(image_data[0])

        result = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                total = 0
                count = 0
                for k in range(-mask // 2, mask // 2 + 1):
                    for l in range(-mask // 2, mask // 2 + 1):
                        if 0 <= i + k < rows and 0 <= j + l < cols:
                            total += image_data[i + k][j + l]
                            count += 1
                result[i][j] = total / count

        return result

    def median_filter(self, image_data, filter_size=3):
        rows = len(image_data)
        cols = len(image_data[0])

        result = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                neighbors = []
                for k in range(-filter_size // 2, filter_size // 2 + 1):
                    for l in range(-filter_size // 2, filter_size // 2 + 1):
                        if 0 <= i + k < rows and 0 <= j + l < cols:
                            neighbors.append(image_data[i + k][j + l])
                result[i][j] = np.median(neighbors)

        return result

    # X = Y / H
    def complex_division(self, y_fourier, h_fourier):
        return np.divide(y_fourier, h_fourier)

    # X = (Y * H*) / (| H | ^ 2 + a ^ 2)
    # H* - комплексно-сопряженный спектр для H
    def complex_noised_division(self, y_fourier, h_fourier, a):
        h_conjugate = np.conjugate(h_fourier)  # H*
        denominator = np.abs(h_fourier) ** 2 + a ** 2  # |H|^2 + a^2

        return np.divide(y_fourier * h_conjugate, denominator)

    def reshape_fourier_big(self, img_data, scale_factor):
        # Прямое 2-D преобразование Фурье
        f_transformed = fft2(img_data)
        f_transformed_shifted = fftshift(f_transformed)

        # Увеличение посредством вставки нулей
        m, n = img_data.shape

        rows_to_add = math.ceil(m * (scale_factor - 1))
        cols_to_add = math.ceil(n * (scale_factor - 1))
        padded = np.pad(f_transformed_shifted,
                        ((rows_to_add // 2, rows_to_add - rows_to_add // 2),
                         (cols_to_add // 2, cols_to_add - cols_to_add // 2)),
                        mode='constant', constant_values=0)

        # Обратное 2-D преобразование Фурье
        f_transformed_shifted_back = ifftshift(padded)
        img_upscaled = ifft2(f_transformed_shifted_back)
        img_upscaled = np.abs(img_upscaled)

        return img_upscaled

    def reshape_fourier_small(self, img_data, scale_factor):
        # Прямое 2-D преобразование Фурье
        f_transformed = fft2(img_data)
        f_transformed_shifted = fftshift(f_transformed)

        m, n = img_data.shape

        # Новые размеры изображения
        new_m = int(m * scale_factor)
        new_n = int(n * scale_factor)

        # Начальные и конечные точки для обрезки
        start_m = (m - new_m) // 2
        start_n = (n - new_n) // 2
        end_m = start_m + new_m
        end_n = start_n + new_n

        # Обрезка частот для уменьшения размера
        trimmed = f_transformed_shifted[start_m:end_m, start_n:end_n]

        # Обратное 2-D преобразование Фурье
        f_transformed_shifted_back = ifftshift(trimmed)
        img_downscaled = ifft2(f_transformed_shifted_back)
        img_downscaled = np.abs(img_downscaled)

        return img_downscaled

    def threshold(self, img_data, limit):
        out_data = np.zeros_like(img_data)

        out_data[img_data >= limit] = 255
        out_data[img_data < limit] = 0

        return out_data

    def dilation(self, img_data, kernel):
        img_h = img_data.shape[0]
        img_w = img_data.shape[1]

        kernel_h, kernel_w = np.shape(kernel)

        kernel_cx = kernel_w // 2
        kernel_cy = kernel_h // 2

        final_image_dilation = np.empty(img_data.shape)

        for row in range(img_h):
            for col in range(img_w):
                max = 0

                for x in range(row - kernel_cx, row + kernel_cx + 1):
                    for y in range(col - kernel_cy, col + kernel_cy + 1):
                        if 0 <= x < img_h and 0 <= y < img_w:
                            if img_data[x, y] > max:
                                max = img_data[x, y]

                final_image_dilation[row, col] = max

        return final_image_dilation

    def erosion(self, img_data, kernel):
        img_h = img_data.shape[0]
        img_w = img_data.shape[1]

        kernel_h, kernel_w = np.shape(kernel)

        kernel_cx = kernel_w // 2
        kernel_cy = kernel_h // 2

        final_image_erosion = np.empty(img_data.shape)

        for row in range(img_h):
            for col in range(img_w):
                min = 255

                for x in range(row - kernel_cx, row + kernel_cy + 1):
                    for y in range(col - kernel_cy, col + kernel_cy + 1):
                        if 0 <= x < img_h and 0 <= y < img_w:
                            if img_data[x, y] < min:
                                min = img_data[x, y]

                final_image_erosion[row, col] = min

        return final_image_erosion
