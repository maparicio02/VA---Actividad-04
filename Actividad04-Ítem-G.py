import cv2
import numpy as np
import os

# ============================================================
# DETECTOR DE BORDES CANNY - PASO A PASO
# ============================================================

def cargar_imagen(ruta="imagen.jpg"):
    """Carga la imagen o crea una de prueba si no existe."""
    if os.path.exists(ruta):
        img = cv2.imread(ruta)
        if img is not None:
            print(f"[OK] Imagen cargada: {ruta}")
            return img

    # Crear imagen de prueba con formas geométricas
    print(f"[INFO] No se encontró '{ruta}'. Creando imagen de prueba...")
    img = np.ones((400, 600, 3), dtype=np.uint8) * 220
    cv2.rectangle(img, (100, 100), (300, 300), (50, 50, 50), -1)
    cv2.circle(img, (450, 200), 80, (30, 30, 30), -1)
    cv2.putText(img, "CANNY", (150, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    cv2.imwrite(ruta, img)
    print(f"[OK] Imagen de prueba guardada: {ruta}")
    return img


def canny_paso_a_paso(img):
    """Muestra cada etapa del detector de Canny."""
    print("\n" + "=" * 60)
    print("   DETECTOR DE BORDES CANNY - PASO A PASO")
    print("=" * 60)

    # Convertir a escala de grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- PASO 1: Suavizado Gaussiano ---
    print("\n[PASO 1] Suavizado Gaussiano (reduce ruido)")
    suavizado = cv2.GaussianBlur(gris, (5, 5), sigmaX=1.4)
    print("         Kernel: 5x5, sigma=1.4")

    # --- PASO 2: Cálculo del Gradiente (Sobel) ---
    print("\n[PASO 2] Cálculo del Gradiente (Sobel)")
    Gx = cv2.Sobel(suavizado, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(suavizado, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitud y dirección
    magnitud = np.sqrt(Gx**2 + Gy**2)
    magnitud = np.uint8(np.clip(magnitud, 0, 255))

    direccion = np.arctan2(Gy, Gx) * 180 / np.pi  # en grados
    print(f"         Magnitud: rango [{magnitud.min()}, {magnitud.max()}]")
    print(f"         Dirección: rango [{direccion.min():.1f}°, {direccion.max():.1f}°]")

    # --- PASO 3: Supresión de No-Máximos (NMS) ---
    print("\n[PASO 3] Supresión de No-Máximos (adelgaza bordes)")
    nms = non_maximum_suppression(magnitud, direccion)
    print("         Bordes adelgazados a 1 píxel")

    # --- PASO 4: Umbralización con Histéresis ---
    print("\n[PASO 4] Umbralización con Histéresis")
    bajo, alto = 30, 90
    print(f"         Umbral bajo = {bajo}, Umbral alto = {alto}")
    bordes = hysteresis(nms, bajo, alto)

    # --- RESULTADO FINAL: Canny de OpenCV (para comparar) ---
    canny_opencv = cv2.Canny(gris, 75, 255)

    # --- MOSTRAR RESULTADOS ---
    cv2.imshow("1. Original", img)
    cv2.imshow("2. Grises", gris)
    cv2.imshow("3. Suavizado Gaussiano", suavizado)
    cv2.imshow("4. Magnitud del Gradiente", magnitud)
    cv2.imshow("5. NMS (adelgazado)", nms)
    cv2.imshow("6. Histéresis (bordes finales)", bordes)
    cv2.imshow("7. Canny OpenCV (referencia)", canny_opencv)

    # Guardar resultados
    cv2.imwrite("canny_1_grises.jpg", gris)
    cv2.imwrite("canny_2_suavizado.jpg", suavizado)
    cv2.imwrite("canny_3_magnitud.jpg", magnitud)
    cv2.imwrite("canny_4_nms.jpg", nms)
    cv2.imwrite("canny_5_bordes.jpg", bordes)
    cv2.imwrite("canny_6_opencv.jpg", canny_opencv)
    print("\n[OK] Resultados guardados como canny_*.jpg")

    print("\n" + "=" * 60)
    print("   Presiona cualquier tecla para cerrar")
    print("=" * 60)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def non_maximum_suppression(magnitud, direccion):
    """Adelgaza los bordes conservando solo máximos locales."""
    filas, cols = magnitud.shape
    nms = np.zeros_like(magnitud)

    # Normalizar dirección a 4 orientaciones: 0°, 45°, 90°, 135°
    angulo = direccion % 180

    for i in range(1, filas - 1):
        for j in range(1, cols - 1):
            q, r = 255, 255  # vecinos a comparar

            # Determinar vecinos según la dirección
            if (0 <= angulo[i, j] < 22.5) or (157.5 <= angulo[i, j] <= 180):
                q, r = magnitud[i, j+1], magnitud[i, j-1]
            elif 22.5 <= angulo[i, j] < 67.5:
                q, r = magnitud[i-1, j+1], magnitud[i+1, j-1]
            elif 67.5 <= angulo[i, j] < 112.5:
                q, r = magnitud[i-1, j], magnitud[i+1, j]
            else:
                q, r = magnitud[i-1, j-1], magnitud[i+1, j+1]

            # Conservar solo si es máximo local
            if magnitud[i, j] >= q and magnitud[i, j] >= r:
                nms[i, j] = magnitud[i, j]

    return nms


def hysteresis(img, bajo, alto):
    """Conserva bordes fuertes y débiles conectados a fuertes."""
    filas, cols = img.shape
    resultado = np.zeros_like(img)

    fuerte = 255
    debil = 75

    # Clasificar píxeles
    for i in range(1, filas - 1):
        for j in range(1, cols - 1):
            if img[i, j] >= alto:
                resultado[i, j] = fuerte
            elif img[i, j] >= bajo:
                resultado[i, j] = debil

    # Propagar bordes débiles conectados a fuertes
    for i in range(1, filas - 1):
        for j in range(1, cols - 1):
            if resultado[i, j] == debil:
                # Verificar si algún vecino es fuerte
                vecindad = resultado[i-1:i+2, j-1:j+2]
                if fuerte in vecindad:
                    resultado[i, j] = fuerte
                else:
                    resultado[i, j] = 0

    return resultado


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("\n=== DETECTOR DE BORDES CANNY PASO A PASO ===")
    print(f"Versión de OpenCV: {cv2.__version__}")

    img = cargar_imagen("prueba.jpg")
    if img is None:
        return

    canny_paso_a_paso(img)


if __name__ == "__main__":
    main()