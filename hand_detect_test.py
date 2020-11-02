''' Ruler 1         2         3         4         5         6         7        '
/*******************************************************************************
*                                                                              *
*                   /$$                                    /$$                 *
*                  | $$                                   | $$                 *
*                  | $$$$$$$    /$$$$$$   /$$$$$$$    /$$$$$$$                 *
*                  | $$__  $$  |____  $$ | $$__  $$  /$$__  $$                 *
*                  | $$  \ $$   /$$$$$$$ | $$  \ $$ | $$  | $$                 *
*                  | $$  | $$  /$$__  $$ | $$  | $$ | $$  | $$                 *
*                  | $$  | $$ |  $$$$$$$ | $$  | $$ |  $$$$$$$                 *
*                  |__/  |__/  \_______/ |__/  |__/  \_______/                 *
*                                                                              *
*              /$$               /$$                             /$$           *
*             | $$              | $$                            | $$           *
*         /$$$$$$$   /$$$$$$   /$$$$$$     /$$$$$$    /$$$$$$$  /$$$$$$        *
*        /$$__  $$  /$$__  $$ |_  $$_/    /$$__  $$  /$$_____/ |_  $$_/        *
*       | $$  | $$ | $$$$$$$$   | $$     | $$$$$$$$ | $$         | $$          *
*       | $$  | $$ | $$_____/   | $$ /$$ | $$_____/ | $$         | $$ /$$      *
*       |  $$$$$$$ |  $$$$$$$   |  $$$$/ |  $$$$$$$ |  $$$$$$$   |  $$$$/      *
*        \_______/  \_______/    \___/    \_______/  \_______/    \___/        *
*                                                                              *
*                  Developed by:                                               *
*                                                                              *
*                            Jhon Hader Fernandez                              *
*                             Diego Fernando Diaz                              *
*                                                                              *
*                  {jhon_fernandez, di-diego}@javeriana.edu.co                 *
*                                                                              *                                                                              *
*                       Pontificia Universidad Javeriana                       *
*                            Bogota DC - Colombia                              *
*                                  Sep - 2020                                  *
*                                                                              *
*****************************************************************************'''

#------------------------------------------------------------------------------#
#                          IMPORT MODULES AND LIBRARIES                        #
#------------------------------------------------------------------------------#

import numpy as np
import imutils
import time
import cv2

# ------------------------------------ main ---------------------------------- #

close_kernel = np.ones((5, 5), np.uint8)
open_kernel = np.ones((5, 5), np.uint8)

bg = None
hand = []

stream = cv2.VideoCapture(0)

print('[INFO...] Getting camera capture. \n')
t = time.time()
while True:

    ret, frame = stream.read()
    if ret == False: break

    # Redimensionar la imagen para que tenga un ancho de 640
    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()

    if bg is not None:

        # Determinar la región de interés (workspace window)
        ROI = frame[50:300, 380:600]
        cv2.rectangle(frame, (380 - 2, 50 - 2), (600 + 2, 300 + 2), (0, 255, 255), 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

        # Región de interés del fondo de la imagen
        bgROI = bg[50:300, 380:600]

        # Determinar la imagen binaria (background vs foreground)
        dif = cv2.absdiff(grayROI, bgROI)
        _, th = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)

        # Closing y opening
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_kernel)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, open_kernel)

        # bordes y contornos
        edges = cv2.Canny(cv2.cvtColor(frame[50:300, 380:600], cv2.COLOR_BGR2GRAY), 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Obtener coordenadas de los contornos
        min_x, max_x, min_y, max_y = [], [], [], []
        is_contours = False
        for k in range(len(contours)):
            min_x.append(min(contours[k][:, 0, 0]))
            max_x.append(max(contours[k][:, 0, 0]))
            min_y.append(min(contours[k][:, 0, 1]))
            max_y.append(max(contours[k][:, 0, 1]))
            is_contours = True

        # Obtener boundary box del mayor contorno
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        if is_contours == True:
            (x_min, y_min) = (min(min_x), min(min_y))
            (x_max, y_max) = (max(max_x), max(max_y))

        drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        hand = ROI

        # Dibujar contornos y boundary box
        for i in range(len(contours)):
            cv2.drawContours(drawing, contours, i, (255, 255, 0))

        cv2.circle(drawing, (x_min, y_min), 8, (0, 0, 255), -1)
        cv2.circle(drawing, (x_max, y_max), 8, (0, 0, 255), -1)
        cv2.rectangle(drawing, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        cv2.circle(hand, (x_min, y_min), 8, (0, 0, 255), -1)
        cv2.circle(hand, (x_max, y_max), 8, (0, 0, 255), -1)
        cv2.rectangle(hand, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        row1 = cv2.hconcat([grayROI, th, opening, edges])
        row2 = cv2.hconcat([drawing, hand])

        cv2.imshow('processing', row1)
        cv2.imshow('hand', row2)
    cv2.imshow('Frame', frame)

    k = cv2.waitKey(10)
    if k == ord('i'):
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
    if k == 27: # escape
        break

stream.release()
cv2.destroyAllWindows()
print('\n[REPORT...] Stream was endend.')
print("[REPORT...] duration of streaming was [s] : {:.3f}".format(time.time() - t))