''' Ruler 1         2         3         4         5         6         7        '
/*******************************************************************************
*                                                                              *
*               ███████╗██╗███╗   ██╗ ██████╗ ███████╗██████╗                  *
*               ██╔════╝██║████╗  ██║██╔════╝ ██╔════╝██╔══██╗                 *
*               █████╗  ██║██╔██╗ ██║██║  ███╗█████╗  ██████╔╝                 *
*               ██╔══╝  ██║██║╚██╗██║██║   ██║██╔══╝  ██╔══██╗                 *
*               ██║     ██║██║ ╚████║╚██████╔╝███████╗██║  ██║                 *
*               ╚═╝     ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝                 *
*                                                                              *
*                 ██████╗ ██████╗ ██╗   ██╗███╗   ██╗████████╗                 *
*                ██╔════╝██╔═══██╗██║   ██║████╗  ██║╚══██╔══╝                 *
*                ██║     ██║   ██║██║   ██║██╔██╗ ██║   ██║                    *
*                ██║     ██║   ██║██║   ██║██║╚██╗██║   ██║                    *
*                ╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║   ██║                    *
*                 ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝   ╚═╝                    *
*                                                                              *
*                  Developed by:                                               *
*                                                                              *
*                            Jhon Hader Fernandez                              *
*                     - jhon_fernandez@javeriana.edu.co                        *
*                                                                              *
*                             Diego Fernando Diaz                              *
*                        - di-diego@javeriana.edu.co                           *
*                                                                              *
*                       Pontificia Universidad Javeriana                       *
*                            Bogota DC - Colombia                              *
*                                  Nov - 2020                                  *
*                                                                              *
*****************************************************************************'''

#------------------------------------------------------------------------------#
#                          IMPORT MODULES AND LIBRARIES                        #
#------------------------------------------------------------------------------#

from sklearn.metrics import pairwise
import numpy as np
import imutils
import cv2


#------------------------------------------------------------------------------#
#                                   FINGER COUNT                               #
#------------------------------------------------------------------------------#

cap = cv2.VideoCapture(0)
bg = None

color_contorno = (255, 0, 255)
color_window = (0, 255, 255)

while True:
    ret, frame = cap.read()
    if ret == False: break

    # Redimensionar la imagen para que tenga un ancho de 640
    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()

    if bg is not None:

        # Determinar la región de interés
        ROI = frame[50:300, 380:600]
        cv2.rectangle(frame, (380 - 2, 50 - 2), (600 + 2, 300 + 2), color_window, 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

        # Región de interés del fondo de la imagen
        bgROI = bg[50:300, 380:600]

        # Determinar la imagen binaria (background vs foreground)
        dif = cv2.absdiff(grayROI, bgROI)
        _, th = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)

        # Opening y closing
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Encontrando los contornos de la imagen binaria
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

        for cnt in cnts:

            # Encontrar el centro del contorno
            M = cv2.moments(cnt)
            if M["m00"] == 0: M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            cv2.circle(ROI, tuple([x, y]), 5, (0, 255, 0), -1)

            # Encontrar boundary box del contorno
            (bnd_x, bnd_y, bnd_w, bnd_h) = cv2.boundingRect(cnt)
            cv2.rectangle(ROI, (bnd_x, bnd_y), (bnd_x+bnd_w, bnd_y+bnd_h), (255, 255, 0), 2)

            # Contorno encontrado a través de cv2.convexHull
            hull = cv2.convexHull(cnt)
            cv2.drawContours(ROI, [hull], 0, color_contorno, 2)

            # Distancias entre maximos puntos y el centro
            top = tuple(hull[hull[:, :, 1].argmin()][0])
            bottom = tuple(hull[hull[:, :, 1].argmax()][0])
            left = tuple(hull[hull[:, :, 0].argmin()][0])
            right = tuple(hull[hull[:, :, 0].argmax()][0])
            dist = pairwise.euclidean_distances([left, right, top], [[x, y]])
            radi = int(0.65 * dist.max())

            # Region de interes circular
            cv2.circle(ROI, (x, y), radi, 255, 6)

            # Mascara de region de interes
            circular_roi = np.zeros(ROI.shape[:-1], dtype=np.uint8)
            cv2.circle(circular_roi, (x, y), radi, 255, 8)

            # Interseccion de mano con region de interes circular
            fingers = cv2.bitwise_and(th, th, mask=circular_roi)

            # Opening
            fingers = cv2.morphologyEx(fingers, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

            # Contornos
            fingers_con, _ = cv2.findContours(fingers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Contar los dedos si el area del contorno cumple con la condicion de area
            finger_count = 0
            for counter in fingers_con:
                if cv2.contourArea(counter) < 300:
                    finger_count += 1

            # Si se cuenta mas de 5 dedos no retorna nulo
            if finger_count > 5:
                finger_count = ' '
            else:
                finger_count = str(finger_count)

            cv2.putText(frame, 'count: ' + finger_count, (460, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), thickness=4)
            cv2.imshow('threshold', th)
            cv2.imshow('fingers', fingers)
    cv2.imshow('Frame', frame)

    k = cv2.waitKey(20)
    if k == ord('i'):
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
