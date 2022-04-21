import cv2
import numpy as np
from PIL import Image


def start_image_object_detection():
    """
    Анализ изображения
    """

    try:
        # Применение методов распознавания объектов на изображении от YOLO
        image = cv2.imread("wRAH1SBIc-A.jpg")
        image = apply_yolo_object_detection(image)

        # Вывод обработанного изображения на экран
        cv2.imshow("Image", image)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        pass


def apply_yolo_object_detection(image_to_process):
    """
    Распознавание и определение координат объектов на изображении
    :param: исходное изображение
    :return: изображение с размеченными объектами и подписями к ним
    """

    height, width, depth = image_to_process.shape
    # Масштабировать и изменить изображение, чтобы оно подходило для нейронной сети
    blob = cv2.dnn.blobFromImage(image_to_process, scalefactor=1/250, size=[608, 608], mean=(0, 0, 0),
                                 swapRB=True, crop=False)
    # Устанавливаем blob как входное значение для сети
    net.setInput(blob)
    # Получаем выходного значения сети
    outs = net.forward(out_layers)

    boxes, class_scores, class_indexes = ([] for _ in range(3))
    objects_count = 0

    # Запуск поиска объектов на изображении
    # Перебираем каждый из выходов слоя
    for out in outs:
        # Перебираем каждое обнаружение объекта
        for obj in out:
            # Извлечение идентификатора класса и достоверность
            # Обнаружение текущего объекта
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            # Достоверность больше 50%
            if class_score > 0.5:
                # Масштабируем координаты ограничивающего прямоугольника относительно размера изображения
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                # Обновляем список координат ограничивающего прямоугольника, достоверности
                # и идентификаторы класса
                box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Удаляем ограничивающие рамки, имеющие достоверность ниже 0.5
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.5)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]

    # Рисуем объекты, входящие в искомый класс
    if classes[class_index] in class_to_look_for:
        objects_count += 1
        image_to_process = draw_object_bounding_box(image_to_process, box)

    return image_to_process


def draw_object_bounding_box(image_to_process, box):
    """
    Обрезание по границам объекта
    :param image_to_process: исходное изображение
    :param box: координаты области вокруг объекта
    :return: изображение, обрезанное по области
    """

    # Координаты ограничивающего прямоугольника
    x, y, w, h = box
    area = (x, y, x + w, y + h)

    # Обрезание изображения по заданной области
    cv2.imwrite('image.jpg', image_to_process, [cv2.IMWRITE_JPEG_QUALITY, 100])
    final_im = Image.open('image.jpg')
    final_image_crop = final_im.crop(area)
    final_image_crop.save('final_image.jpg', quality=95)
    result_image = cv2.imread("final_image.jpg")

    return result_image


if __name__ == '__main__':
    # Загрузка весов YOLO из файлов и настройка сети
    net = cv2.dnn.readNetFromDarknet("yolov4-csp.cfg", "yolov4-csp.weights")

    # Получаем все слои
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # Загрузка из файла классов объектов, которые умеет обнаруживать YOLO
    with open("coco.names.txt") as file:
        classes = file.read().split("\n")

    # Определение класса, поиск которого будет осуществляться
    class_to_look_for = "cat"

    start_image_object_detection()
