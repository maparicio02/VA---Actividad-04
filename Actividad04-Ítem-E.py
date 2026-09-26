"""
================================================================================
 ESCANER DE DOCUMENTOS + OCR -- VERSION 2 (mejorada, para imagenes REALES)
================================================================================
Mejoras respecto a la v1:

  1. Auto-Canny: los umbrales de Canny ya no son fijos (75,200); se calculan
     a partir de la mediana de intensidad de CADA imagen (metodo de
     Rosebrock). Esto es clave para generalizar a fotos reales, que varian
     mucho en exposicion/contraste entre si.
  2. CLAHE (ecualizacion de histograma adaptativa) antes de Canny, para que
     los bordes se detecten tambien en zonas de sombra o bajo contraste
     (ej. papel blanco sobre mesa clara).
  3. Denoising (filtro bilateral) para limpiar el grano/ruido real de fotos
     de celular antes de calcular gradientes.
  4. Deteccion robusta del documento en DOS niveles:
       a) contorno de 4 vertices (approxPolyDP) sobre el mapa de Canny
       b) si falla, minAreaRect del contorno mas grande (rectangulo
          rotado), que tolera esquinas redondeadas o papel curvado
       c) si Canny tampoco da un area razonable (bordes con huecos, muy
         comun en papel curvado o fotos reales), se usa una MASCARA POR
         THRESHOLDING (Otsu) como fuente alternativa de contorno: a
         diferencia de Canny (linea de 1px que se puede cortar por huecos),
         Otsu entrega una REGION RELLENA, mucho mas tolerante a huecos.
         Esta es la combinacion practica Canny + thresholding.
      d) si todo falla, usa la imagen completa (fallback controlado)
  5. Binarizacion MULTI-ESTRATEGIA: en vez de aplicar un solo umbral fijo,
     se prueban 4 variantes (gris plano, Otsu, umbral adaptativo medio,
     umbral adaptativo gaussiano) y se corre OCR de verdad sobre cada una,
     midiendo la confianza promedio que reporta Tesseract
     (pytesseract.image_to_data). Se elige automaticamente la variante que
     dio mejor confianza -- esto es lo que en un dashboard de OCR real se
     llama "auto-tuning" del preprocesamiento.

Uso:
    python scanner_ocr_v2.py imagen.jpg --salida salida --lang eng
    python scanner_ocr_v2.py imagen.jpg --lang spa+eng   # multi-idioma
================================================================================
"""

import os
import argparse
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Configuracion robusta de Tesseract para Windows.
# Si el binario estara en una ruta no estandar, se usa automaticamente.
possible_tesseract_paths = [
    r"C:\\msys64\\ucrt64\\bin\\tesseract.exe",
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
    r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
    r"C:\\Users\\HP\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe",
]
for candidate in possible_tesseract_paths:
    if os.path.exists(candidate):
        pytesseract.pytesseract.tesseract_cmd = candidate
        break

# Si Tesseract no tiene el directorio de datos correcto, se fuerza la ruta real
# de la instalacion de MSYS2 para que encuentre eng.traineddata y spa.traineddata.
for tessdata_dir in [
    r"C:\msys64\ucrt64\share\tessdata",
    r"C:\Program Files\Tesseract-OCR\tessdata",
    r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
    r"C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tessdata",
]:
    if os.path.isdir(tessdata_dir) and os.path.exists(os.path.join(tessdata_dir, "eng.traineddata")):
        os.environ["TESSDATA_PREFIX"] = tessdata_dir
        break

# Si la ruta de datos aun no existe, pero la que usa Tesseract si existe, se deja como referencia.
if not os.environ.get("TESSDATA_PREFIX") and os.path.isdir(r"C:\msys64\ucrt64\share\tessdata"):
    os.environ["TESSDATA_PREFIX"] = r"C:\msys64\ucrt64\share\tessdata"


# --------------------------------------------------------------------------
# GEOMETRIA
# --------------------------------------------------------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    width, height = max(width, 1), max(height, 1)

    dst = np.array([[0, 0], [width - 1, 0],
                     [width - 1, height - 1], [0, height - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (width, height))


# --------------------------------------------------------------------------
# MEJORA 1: AUTO-CANNY (umbrales adaptados a cada imagen, no fijos)
# --------------------------------------------------------------------------
def auto_canny(gray, sigma=0.33):
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


# --------------------------------------------------------------------------
# DETECCION DEL DOCUMENTO (con fallback de 3 niveles)
# --------------------------------------------------------------------------
def find_document_points(image, debug_dir=None, tag=""):
    ratio = image.shape[0] / 700.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 700))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # MEJORA 2: CLAHE -> realza contraste local antes de buscar bordes
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # MEJORA 3: denoising bilateral (preserva bordes, limpia textura/ruido)
    denoised = cv2.bilateralFilter(gray_eq, 9, 75, 75)

    edged = auto_canny(denoised)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{tag}01_clahe.png"), gray_eq)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}02_canny.png"), edged)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    img_area = resized.shape[0] * resized.shape[1]

    # Nivel A: buscar un contorno de exactamente 4 vertices
    for c in cnts:
        if cv2.contourArea(c) < 0.15 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2) * ratio, "Canny + 4 vertices"

    # Nivel B: minAreaRect del contorno mas grande (tolera esquinas
    # redondeadas / papel curvado, como en fotos de recibos reales)
    if cnts and cv2.contourArea(cnts[0]) > 0.15 * img_area:
        rect = cv2.minAreaRect(cnts[0])
        box = cv2.boxPoints(rect)
        return box * ratio, "Canny + minAreaRect (fallback)"

    # Nivel C: MASCARA POR THRESHOLDING (Otsu) como respaldo de Canny.
    # Se prueban las dos polaridades (documento claro sobre fondo oscuro,
    # o viceversa) y se toma la que produzca un blob grande y solido.
    mejor_c = None
    for flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, mask = cv2.threshold(denoised, 0, 255, flag + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mcnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not mcnts:
            continue
        c = max(mcnts, key=cv2.contourArea)
        area_frac = cv2.contourArea(c) / img_area
        # el documento no deberia ser casi toda la imagen (eso indicaria
        # que se selecciono el fondo, no el papel) ni demasiado chico
        if 0.15 < area_frac < 0.92:
            if mejor_c is None or area_frac > mejor_c[1]:
                mejor_c = (c, area_frac)

    if mejor_c is not None:
        c = mejor_c[0]
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2) * ratio, "Otsu (mascara) + 4 vertices"
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        return box * ratio, "Otsu (mascara) + minAreaRect"

    # Nivel D: no se encontro nada confiable
    return None, "Sin recorte (imagen completa)"


# --------------------------------------------------------------------------
# MEJORA 5: BINARIZACION MULTI-ESTRATEGIA CON SELECCION POR CONFIANZA OCR
# --------------------------------------------------------------------------
def _mean_confidence(gray_or_binary_img, lang):
    data = pytesseract.image_to_data(gray_or_binary_img, lang=lang, output_type=Output.DICT)
    confs = [int(c) for c in data["conf"] if c not in ("-1", -1)]
    return (sum(confs) / len(confs)) if confs else -1.0


def best_binarization(warped_bgr, lang="eng", debug_dir=None):
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    candidatos = {
        "gris_plano": gray,
        "otsu": cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        "adaptativo_medio": cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 15),
        "adaptativo_gauss": cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15),
    }

    resultados = {}
    for nombre, img in candidatos.items():
        try:
            conf = _mean_confidence(img, lang)
        except Exception:
            conf = -1.0
        resultados[nombre] = conf
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"bin_{nombre}.png"), img)

    mejor = max(resultados, key=resultados.get)
    return candidatos[mejor], mejor, resultados


# --------------------------------------------------------------------------
# PIPELINE COMPLETO
# --------------------------------------------------------------------------
def scan_and_ocr(image_path, output_dir="salida", lang="eng", debug=True):
    """
    MEJORA FINAL: auto-verificacion por confianza de OCR.
    Ninguna heuristica geometrica (ni Canny ni Otsu) es infalible -- en
    imagenes de muy bajo contraste puede recortar un pedazo de fondo en
    vez del documento. Por eso se generan DOS candidatos --el recorte
    geometrico y la imagen completa-- se le corre el binarizador
    multi-estrategia (que ya elige el mejor umbral) a cada uno, y se
    escoge el candidato con mayor confianza real de Tesseract. Asi el
    propio OCR actua como "juez" final de todo el pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    dbg = output_dir if debug else None
    pts, metodo_geo = find_document_points(image, debug_dir=dbg)

    candidatos = [("imagen completa", image)]
    if pts is not None:
        candidatos.append((metodo_geo, four_point_transform(image, pts)))

    mejor = None
    for i, (nombre_cand, warped) in enumerate(candidatos):
        binaria, metodo_bin, confs = best_binarization(warped, lang=lang)
        conf = confs[metodo_bin]
        if dbg:
            tag = "completa" if i == 0 else "recorte"
            cv2.imwrite(os.path.join(output_dir, f"candidato_{tag}.png"), warped)
        if mejor is None or conf > mejor["confianza_elegida"]:
            mejor = {
                "metodo_deteccion": nombre_cand,
                "metodo_binarizado": metodo_bin,
                "confianzas": confs,
                "confianza_elegida": conf,
                "warped": warped,
                "binaria": binaria,
            }

    if dbg:
        cv2.imwrite(os.path.join(output_dir, "03_rectificado_elegido.png"), mejor["warped"])
        cv2.imwrite(os.path.join(output_dir, "04_binarizado_final.png"), mejor["binaria"])

    texto = pytesseract.image_to_string(mejor["binaria"], lang=lang)
    with open(os.path.join(output_dir, "texto_extraido.txt"), "w", encoding="utf-8") as f:
        f.write(texto)

    return {
        "metodo_deteccion": mejor["metodo_deteccion"],
        "metodo_binarizado": mejor["metodo_binarizado"],
        "confianzas": mejor["confianzas"],
        "confianza_elegida": mejor["confianza_elegida"],
        "texto": texto.strip(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Escaner de documentos + OCR v2 (mejorado)")
    parser.add_argument("imagen", help="Ruta a la imagen del documento")
    parser.add_argument("--salida", default="salida", help="Carpeta de salida")
    parser.add_argument("--lang", default="eng", help="Idioma(s) para Tesseract, ej. eng, spa, spa+eng")
    args = parser.parse_args()

    r = scan_and_ocr(args.imagen, args.salida, args.lang)
    print(f"Deteccion del documento : {r['metodo_deteccion']}")
    print(f"Mejor binarizacion      : {r['metodo_binarizado']}  (confianza {r['confianza_elegida']:.1f})")
    print(f"Confianzas por metodo   : {r['confianzas']}")
    print("----- TEXTO EXTRAIDO -----")
    print(r["texto"])