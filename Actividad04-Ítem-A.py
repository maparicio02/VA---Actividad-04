import cv2
import numpy as np

# ============================================================
# OCR CON THRESHOLDING: Otsu (Global) vs Adaptativo (Local)
# ============================================================

def crear_imagen_prueba():
    """Crea una imagen de prueba con texto e iluminación desigual."""
    img = np.ones((300, 800), dtype=np.uint8) * 240
    cv2.putText(img, "OPENCV OCR TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)
    cv2.putText(img, "Otsu vs Adaptativo", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    cv2.putText(img, "1234567890", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    # Degradado de iluminación (simula sombra)
    for i in range(img.shape[1]):
        img[:, i] = (img[:, i] * (1 - i / img.shape[1] * 0.6)).astype(np.uint8)
    return img


def evaluar(binaria, nombre):
    """Cuenta letras (componentes grandes) y ruido (componentes pequeños)."""
    inv = cv2.bitwise_not(binaria)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    letras = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20)
    ruido  = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < 5)
    print(f"   [{nombre}] Letras detectadas: {letras} | Ruido: {ruido}")
    return letras, ruido


def main():
    # 1. Cargar imagen (o crear una de prueba)
    img = cv2.imread("Argentina2.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("[INFO] No se encontró 'Argentina.jpg'. Usando imagen de prueba.")
        img = crear_imagen_prueba()

    # 2. Aplicar los dos tipos de threshold
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt   = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)

    # 3. Invertir si es necesario (texto negro sobre fondo blanco para OCR)
    for nombre, b in [("Otsu", otsu), ("Adaptativo", adapt)]:
        if np.sum(b == 255) < np.sum(b == 0):
            if nombre == "Otsu": otsu = cv2.bitwise_not(b)
            else: adapt = cv2.bitwise_not(b)

    # 4. Evaluar calidad
    print("\n--- MÉTRICAS ---")
    l_otsu, r_otsu = evaluar(otsu, "Otsu")
    l_adapt, r_adapt = evaluar(adapt, "Adaptativo")

    # 5. Mostrar resultados
    cv2.imshow("Original", img)
    cv2.imshow("Otsu (Global)", otsu)
    cv2.imshow("Adaptativo (Local)", adapt)
    cv2.imwrite("resultado_otsu.jpg", otsu)
    cv2.imwrite("resultado_adaptativo.jpg", adapt)

    # 6. Conclusión
    print("\n--- CONCLUSIÓN ---")
    if l_adapt >= l_otsu and r_adapt <= r_otsu:
        print("Mejor: ADAPTATIVO (Local) → más letras y menos ruido.")
    elif l_otsu > l_adapt:
        print("Mejor: OTSU (Global) → más letras detectadas.")
    else:
        print("Mejor: ADAPTATIVO (Local) → más robusto ante iluminación desigual.")

    print("""
Explicación:
 • Otsu calcula UN umbral global desde el histograma.
   Funciona bien con iluminación uniforme, pero falla con sombras.
 • Adaptativo calcula un umbral por región (ventana local).
   Se adapta a cambios de iluminación → mejor en fotos reales.
 • Para OCR: si el documento está bien escaneado → Otsu.
   Si es foto con luz irregular → Adaptativo.
    """)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()