import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import imutils

from classes.model import Model
from classes.in_out import In_Out
from classes.analysis import Analysis
from classes.processing import Processing

model = Model()
in_out = In_Out()
analysis = Analysis()
processing = Processing()

image_name = 'stones'
is_color = False
stone_size = 8


def highlight_outlines(image, labels, markers, size):
    img_copy = image.copy()
    stones = 0

    for label in labels[2:]:
        # Создание бинарного изображения
        target = np.where(markers == label, 255, 0).astype(np.uint8)

        # Поиск контуров на бинарном изображении
        outlines = cv2.findContours(target.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines = imutils.grab_contours(outlines)

        if len(outlines) > 0:
            # Вычисление ограничивающего прямоугольника для самого большого контура
            biggest_outline = max(outlines, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(biggest_outline)

            # Отрисовка прямоугольника и увеличение счетчика
            if (int(w) < size and int(h) == size) or (int(h) < size and int(w) == size):
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 1)
                stones += 1

    print(f'Number of stones found: {stones}')
    return img_copy


def print_bounding_boxes(binary_image, original_image):
    # Поиск контуров на бинарном изображении
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создание ограничивающих прямоугольников для каждого контура
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    # Отрисовка ограничивающих прямоугольников на исходном изображении
    image_with_boxes = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(image_with_boxes, f"{w}x{h}", (x + 3, y + h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    return image_with_boxes


def imshow(img, ax):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')


def coins(img, labels, markers):
    coins = []

    for label in labels[2:]:
        # Создание бинарного изображения, на котором
        # область метки находится на переднем плане,
        # а остальная часть изображения - на заднем
        target = np.where(markers == label, 255, 0).astype(np.uint8)

        # Извлечение контуров бинарного изображения
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        coins.append(contours[0])

    # Отрисовка контуров
    img = cv2.drawContours(img, coins, -1, color=(0, 0, 255), thickness=1)

    plt.figure()
    plt.axis('off')
    plt.title('outline')
    plt.imshow(img)
    plt.show()


def vv(bin_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    # Уверенный фон
    sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
    imshow(sure_bg, axes[0, 0])
    axes[0, 0].set_title('Sure Background')

    # Преобразование расстояния
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)
    imshow(dist, axes[0, 1])
    axes[0, 1].set_title('Distance Transform')

    # Область переднего плана
    ret, sure_fg = cv2.threshold(dist, 0.07 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    imshow(sure_fg, axes[1, 0])
    axes[1, 0].set_title('Sure Foreground')

    # Неизвестная область
    unknown = cv2.subtract(sure_bg, sure_fg)
    imshow(unknown, axes[1, 1])
    axes[1, 1].set_title('Unknown')

    plt.show()

    cv2.imwrite('data/jpg/' + image_name + '/' + image_name + '_bg.jpg', sure_bg)
    cv2.imwrite('data/jpg/' + image_name + '/' + image_name + '_fg.jpg', sure_fg)
    cv2.imwrite('data/jpg/' + image_name + '/' + image_name + '_dist.jpg', dist)
    cv2.imwrite('data/jpg/' + image_name + '/' + image_name + '_unknown.jpg', unknown)

    return sure_fg, unknown


def get_markers(sure_fg, unknown):
    # Маркирование уверенного переднего плана
    ret, markers = cv2.connectedComponents(sure_fg)

    # Добавление 1 ко всем меткам, чтобы фон был не 0, а 1.
    markers += 1
    # Неизвестная область = 0
    markers[unknown == 255] = 0

    return markers


def watershed(img, markers):
    # Алгоритм водораздела
    markers = cv2.watershed(img, markers)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(markers, cmap="tab20b")
    ax.axis('off')
    plt.show()

    cv2.imwrite('data/jpg/' + image_name + '/' + image_name + '_markers.jpg', markers)
    labels = np.unique(markers)
    return labels


def fill_contour_area(binary_image):
    # Копирование бинарного изображения
    filled_image = binary_image.copy()

    # Нахождение контуров на бинарном изображении
    contours, hierarchy = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Заполнение области внутри каждого контура
    for contour in contours:
        cv2.fillPoly(filled_image, pts=[contour], color=255)

    return filled_image


def main():
    kernel_size = 3

    # Image loading
    img = cv2.imread('data/jpg/' + image_name + '/' + image_name + '.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, bin_img = cv2.threshold(gray,
                                 0, 255,
                                 cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    filtered = cv2.morphologyEx(bin_img,
                                cv2.MORPH_OPEN,
                                kernel,
                                iterations=1)

    in_out.show_jpg_files([img, bin_img, filtered],
                          ['original', 'binary image', 'filtered image'],
                          is_color)

    sure_fg, unknowm = vv(filtered)
    markers = get_markers(sure_fg, unknowm)
    labels = watershed(img, markers)
    img2 = highlight_outlines(img, labels, markers, stone_size)

    in_out.show_jpg_sub(img2, is_color, 'highlight outlines')
    coins(img, labels, markers)

    image_with_all_boxes = print_bounding_boxes(filtered, gray)
    in_out.show_jpg_sub(image_with_all_boxes, is_color, 'image with boxes outlines')

    cv2.imwrite('data/jpg/' + image_name + '/' + image_name + '_bin.jpg', bin_img)
    cv2.imwrite('data/jpg/' + image_name + '/' + image_name + '_filtered.jpg', filtered)
    cv2.imwrite('data/jpg/' + image_name + '/' + image_name + '_found.jpg', img2)
    cv2.imwrite('data/jpg/' + image_name + '/' + image_name + '_with_sizes.jpg', image_with_all_boxes)
    cv2.imwrite('data/jpg/' + image_name + '/' + image_name + '_contours.jpg', img)
