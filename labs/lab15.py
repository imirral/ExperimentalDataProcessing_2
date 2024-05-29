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

image_path = 'data/jpg/stones/'
image_name = 'stones'
is_color = False
stone_size = 8


def main():
    kernel_size = 3

    # Считывание изображения
    image_data = cv2.imread(image_path + image_name + '.jpg')
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    filtered_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)

    in_out.show_jpg_files([image_data, binary_image, filtered_image],
                          ['original image', 'binary image', 'filtered image'],
                          is_color)

    foreground, unknown = get_image_areas(filtered_image)
    markers = get_markers(foreground, unknown)
    labels = watershed(image_data, markers)

    image_with_stones_outlines = highlight_stones_outlines(image_data, labels, markers)
    image_with_all_boxes = highlight_all_boxes(filtered_image, gray_image)
    image_with_suitable_boxes = highlight_suitable_boxes(image_data, labels, markers)

    in_out.show_jpg_sub(image_with_stones_outlines, is_color, 'image with stones outlines')
    in_out.show_jpg_sub(image_with_all_boxes, is_color, 'image with all boxes outlines')
    in_out.show_jpg_sub(image_with_suitable_boxes, is_color, 'image with suitable boxes outlines')

    cv2.imwrite(image_path + image_name + '_binary.jpg', binary_image)
    cv2.imwrite(image_path + image_name + '_filtered.jpg', filtered_image)
    cv2.imwrite(image_path + image_name + '_outlines.jpg', image_with_stones_outlines)
    cv2.imwrite(image_path + image_name + '_suitable_boxes.jpg', image_with_suitable_boxes)
    cv2.imwrite(image_path + image_name + '_all_boxes.jpg', image_with_all_boxes)


def get_image_areas(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    fig, axes = plt.subplots(nrows=2, ncols=2)

    # Область заднего плана
    background = cv2.dilate(binary_image, kernel, iterations=3)
    in_out.imshow(background, axes[0, 0])
    axes[0, 0].set_title('image\'s background')

    # Расстояние между каждым ненулевым и ближайшим нулевым пикселями
    distance = cv2.distanceTransform(binary_image, cv2.DIST_L2, 3)
    in_out.imshow(distance, axes[0, 1])
    axes[0, 1].set_title('distance transform')

    # Область переднего плана
    _, foreground = cv2.threshold(distance, 0.07 * distance.max(), 255, cv2.THRESH_BINARY)
    foreground = foreground.astype(np.uint8)
    in_out.imshow(foreground, axes[1, 0])
    axes[1, 0].set_title('image\'s foreground')

    # Неизвестная область
    unknown = cv2.subtract(background, foreground)
    in_out.imshow(unknown, axes[1, 1])
    axes[1, 1].set_title('unknown area')

    plt.show()

    cv2.imwrite(image_path + image_name + '_background.jpg', background)
    cv2.imwrite(image_path + image_name + '_foreground.jpg', foreground)
    cv2.imwrite(image_path + image_name + '_distance.jpg', distance)
    cv2.imwrite(image_path + image_name + '_unknown.jpg', unknown)

    return foreground, unknown


def get_markers(foreground, unknown):
    # Маркирование переднего плана (извлечение области)
    _, markers = cv2.connectedComponents(foreground)

    # Извлеченная область = 1
    markers += 1
    # Неизвестная область = 0
    markers[unknown == 255] = 0

    return markers


def watershed(image_data, markers):
    # Алгоритм водораздела
    markers = cv2.watershed(image_data, markers)

    fig, axes = plt.subplots(figsize=(5, 5))
    axes.imshow(markers, cmap='gray')
    axes.axis('off')
    plt.show()

    cv2.imwrite(image_path + image_name + '_markers.jpg', markers)
    labels = np.unique(markers)

    return labels


def highlight_stones_outlines(image_data, labels, markers):
    image_with_outlines = image_data.copy()

    outlines = []

    for label in labels[2:]:
        # Создание бинарного изображения
        target = np.where(markers == label, 255, 0).astype(np.uint8)

        # Извлечение контуров бинарного изображения
        contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines.append(contours[0])

    # Отрисовка контуров
    image_with_outlines = cv2.drawContours(image_with_outlines, outlines, -1, color=(0, 0, 255), thickness=1)

    return image_with_outlines


def highlight_all_boxes(binary_image, original_image):
    # Поиск контуров на бинарном изображении
    outlines, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создание ограничивающих прямоугольников для каждого контура
    bounding_boxes = []

    for outline in outlines:
        x, y, w, h = cv2.boundingRect(outline)
        bounding_boxes.append((x, y, w, h))

    # Отрисовка ограничивающих прямоугольников на исходном изображении
    image_with_boxes = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 1)

    return image_with_boxes


def highlight_suitable_boxes(image_data, labels, markers):
    image_with_boxes = image_data.copy()

    square_stones = 0
    rect_stones = 0

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

            # Если обе стороны прямоугольника == stone_size
            if int(w) == stone_size and int(h) == stone_size:
                # Отрисовка прямоугольника и увеличение счетчика
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 1)
                square_stones += 1

            # Если одна из сторон прямоугольника == stone_size, а другая сторона < stone_size
            if (int(w) < stone_size and int(h) == stone_size) or (int(h) < stone_size and int(w) == stone_size):
                # Отрисовка прямоугольника и увеличение счетчика
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 1)
                rect_stones += 1

    print(f'Number of stones (height = {stone_size}, width = {stone_size}): {square_stones}')
    print(f'Number of stones (height <= {stone_size}, width <= {stone_size}: {rect_stones}')

    return image_with_boxes
