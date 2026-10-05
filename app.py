import streamlit as st
import pandas as pd
import json
import unicodedata
import io
from datetime import datetime, date, timedelta
from pathlib import Path
from difflib import SequenceMatcher

from fpdf import FPDF

try:
    import qrcode
    QR_DISPONIBLE = True
except ImportError:
    QR_DISPONIBLE = False

try:
    import cv2
    import numpy as np
    CV_DISPONIBLE = True
except ImportError:
    CV_DISPONIBLE = False

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from PIL import Image as PILImage
    PYZBAR_DISPONIBLE = True
except ImportError:
    PYZBAR_DISPONIBLE = False


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
TOTAL_ASIENTOS = 440
UMBRAL_SIMILITUD = 0.80

CARPETA_DATOS = Path("datos")
CARPETA_DATOS.mkdir(exist_ok=True)

ARCHIVO_VENTAS = CARPETA_DATOS / "ventas_boletos.csv"
ARCHIVO_CONFIG = CARPETA_DATOS / "config.json"
ARCHIVO_LOGO = Path("logo.png")

VENDEDORES = ["Isaac", "Sara", "Ximena", "GENERAL"]
MODOS_PAGO = ["Efectivo", "Transferencia SPEI"]

MODALIDADES = {
    "Entrada General": {
        "precio": 60.00, "cantidad": 1,
        "nota": "Boleto individual a precio general.",
    },
    "Preventa": {
        "precio": 50.00, "cantidad": 1,
        "nota": "Boleto individual a precio de preventa.",
    },
    "Promoción 2 x \\$100": {
        "precio": 50.00, "cantidad": 2,
        "nota": "Dos boletos por $100 (ahorro de $20 vs. precio general).",
    },
}

COLUMNAS = [
    "Folio", "Nombre", "Costo del boleto", "Cantidad de boletos",
    "Modo de pago", "Vendido por:", "Fecha",
    "Teléfono de contacto", "e-mail", "Boleto digital",
    "Circunstancia", "Comentarios", "Estado", "Check-in",
]

ESTADOS_BOLETO = ["Válido", "Usado", "Cancelado"]

OBRA_TITULO = "Las Preciosas Ridículas"
OBRA_AUTOR = "Adaptación de la obra de Molière"
OBRA_LUGAR = 'Foro Cultural Tiempo y Espacio "Thaay" · Lerma, Estado de México'
OBRA_FECHA = "Sábado 19 de diciembre · 12:00 hrs"
OBRA_GRUPO = "Taller de Teatro · Grupo Matutino"

SPEI_BANCO = "Nubank"
SPEI_BENEFICIARIO = "Isaac Lael Gómez Fol"
SPEI_CLABE = "6381 8000 0113 9816 25"
SPEI_CONCEPTO = "NÚMERODEBOLETOS+TUNOMBRE"


# ============================================================
# HELPERS DE FOLIOS
# ============================================================
def folios_usados(df):
    if df.empty:
        return set()
    usados = set()
    for f in df["Folio"]:
        try:
            fi = int(f)
            if fi > 0:
                usados.add(fi)
        except (ValueError, TypeError):
            pass
    return usados


def folios_compact(folios):
    if not folios:
        return "—"
    folios = sorted(folios)
    if len(folios) == 1:
        return str(folios[0])
    if folios == list(range(folios[0], folios[0] + len(folios))):
        return f"{folios[0]}–{folios[-1]}"
    return ", ".join(str(f) for f in folios)


def asignar_folios(df, cantidad, next_folio):
    usados = folios_usados(df)
    cantidad = int(cantidad)
    consecutivos = []
    for i in range(cantidad):
        f = next_folio + i
        if f > TOTAL_ASIENTOS or f in usados:
            consecutivos = []
            break
        consecutivos.append(f)
    if len(consecutivos) == cantidad:
        return consecutivos
    libres = [f for f in range(1, TOTAL_ASIENTOS + 1) if f not in usados]
    if len(libres) >= cantidad:
        return libres[:cantidad]
    return None


def recalcular_next_folio(df, cfg):
    usados = folios_usados(df)
    cfg["next_folio"] = 1 if not usados else max(usados) + 1
    guardar_config(cfg)
    return cfg


# ============================================================
# HELPERS GENERALES
# ============================================================
def normalizar(texto):
    t = str(texto).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return " ".join(t.split())


def precio_fmt(v):
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _es_si(v):
    return str(v).strip().lower() in ("sí", "si", "yes", "1", "true", "x")


# ============================================================
# LECTURA DE QR Y VALIDACIÓN DE BOLETOS
# ============================================================
def leer_qr_desde_bytes(bytes_imagen):
    """
    Intenta leer un QR desde los bytes de una imagen.
    Estrategia: pyzbar primero (más robusto), luego OpenCV como respaldo.
    Devuelve el texto del QR o None.
    """
    # --- Vía 1: pyzbar (más robusto en QRs borrosos) ---
    if PYZBAR_DISPONIBLE:
        try:
            img = PILImage.open(io.BytesIO(bytes_imagen))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            resultados = pyzbar_decode(img)
            for r in resultados:
                if r.type == "QRCODE":
                    return r.data.decode("utf-8", errors="replace")
            # Si no hubo QRCODE, pero sí otro código, devolvemos el primero
            if resultados:
                return resultados[0].data.decode("utf-8", errors="replace")
        except Exception:
            pass

    # --- Vía 2: OpenCV (fallback) ---
    if CV_DISPONIBLE:
        try:
            nparr = np.frombuffer(bytes_imagen, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                detector = cv2.QRCodeDetector()
                data, _, _ = detector.detectAndDecode(img)
                if data:
                    return data
        except Exception:
            pass

    return None


def extraer_folio_de_qr(texto_qr):
    """Del texto 'THAAY|FOLIO:X|NOMBRE:...|FECHA:...' extrae el folio (int)."""
    if not texto_qr:
        return None
    for parte in str(texto_qr).split("|"):
        if parte.strip().upper().startswith("FOLIO:"):
            try:
                return int(parte.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                return None
    return None


def validar_folio(df, folio):
    """Devuelve un dict con el resultado de la validación."""
    resultado = {
        "existe": False, "estado": "", "check_in": "",
        "nombre": "", "fila_idx": None,
        "fecha": "", "modalidad": "", "vendido_por": "",
    }
    if df.empty or folio is None:
        return resultado
    coincidencias = df.index[df["Folio"].astype(int) == int(folio)].tolist()
    if not coincidencias:
        return resultado
    idx = coincidencias[0]
    fila = df.loc[idx]
    resultado["existe"] = True
    resultado["estado"] = str(fila.get("Estado", "Válido")).strip() or "Válido"
    resultado["check_in"] = str(fila.get("Check-in", "")).strip()
    resultado["nombre"] = str(fila.get("Nombre", ""))
    resultado["fila_idx"] = idx
    resultado["fecha"] = str(fila.get("Fecha", ""))
    resultado["modalidad"] = str(fila.get("Circunstancia", ""))
    resultado["vendido_por"] = str(fila.get("Vendido por:", ""))
    return resultado


def marcar_boleto_usado(df, folio):
    """Marca un folio como Usado con timestamp y guarda el CSV."""
    idx = df.index[df["Folio"].astype(int) == int(folio)].tolist()
    if not idx:
        return df
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[idx[0], "Estado"] = "Usado"
    df.loc[idx[0], "Check-in"] = ahora
    guardar_ventas(df)
    return df


def cambiar_estado_boleto(df, folio, nuevo_estado):
    """Cambia el estado de un boleto (Válido/Usado/Cancelado)."""
    idx = df.index[df["Folio"].astype(int) == int(folio)].tolist()
    if not idx:
        return df
    df.loc[idx[0], "Estado"] = nuevo_estado
    if nuevo_estado != "Usado":
        df.loc[idx[0], "Check-in"] = ""
    guardar_ventas(df)
    return df


# ============================================================
# FUENTES PARA PDF
# ============================================================
def _buscar_fuentes_dejavu():
    bases = [
        Path("/usr/share/fonts/dejavu-sans-fonts"),
        Path("/usr/share/fonts/dejavu"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/TTF"),
        Path("."),
        Path("fuentes"),
    ]

    def _find(nombre):
        for base in bases:
            p = base / nombre
            if p.exists():
                return p
        return None

    regular = _find("DejaVuSans.ttf")
    bold = _find("DejaVuSans-Bold.ttf") or regular
    italic = _find("DejaVuSans-Oblique.ttf") or regular
    bold_italic = _find("DejaVuSans-BoldOblique.ttf") or bold or italic or regular

    if not regular:
        return None

    return {"": regular, "B": bold, "I": italic, "BI": bold_italic}


# ============================================================
# GENERACIÓN DE BOLETO PDF CON QR
# ============================================================
def generar_boleto_pdf(datos_venta):
    """Genera un boleto PDF con diseño elaborado y código QR. Devuelve bytes."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    fuentes = _buscar_fuentes_dejavu()
    if fuentes:
        try:
            pdf.add_font("Fuente", "", str(fuentes[""]))
            pdf.add_font("Fuente", "B", str(fuentes["B"]))
            pdf.add_font("Fuente", "I", str(fuentes["I"]))
            pdf.add_font("Fuente", "BI", str(fuentes["BI"]))
            FUENTE = "Fuente"
        except Exception:
            FUENTE = "Helvetica"
    else:
        FUENTE = "Helvetica"

    dorado = (184, 134, 11)
    marron = (62, 39, 35)
    crema = (245, 240, 232)
    gris = (100, 100, 100)
    dorado_suave = (215, 180, 90)

    # --- ENCABEZADO ---
    pdf.set_fill_color(*crema)
    pdf.rect(0, 0, 210, 95, style="F")

    if ARCHIVO_LOGO.exists():
        try:
            pdf.image(str(ARCHIVO_LOGO), x=80, y=10, w=50)
            pdf.set_y(64)
        except Exception:
            pdf.set_y(20)
    else:
        pdf.set_y(20)

    pdf.set_font(FUENTE, "B", 26)
    pdf.set_text_color(*dorado)
    pdf.cell(0, 14, OBRA_TITULO, ln=True, align="C")

    pdf.set_font(FUENTE, "I", 12)
    pdf.set_text_color(*marron)
    pdf.cell(0, 8, OBRA_AUTOR, ln=True, align="C")
    pdf.ln(3)

    pdf.set_draw_color(*dorado)
    pdf.set_line_width(0.6)
    y_line = pdf.get_y()
    pdf.line(60, y_line, 95, y_line)
    pdf.line(115, y_line, 150, y_line)
    pdf.set_fill_color(*dorado)
    cx, cy, s = 105, y_line, 1.8
    pdf.polygon(
        [(cx - s, cy), (cx, cy - s), (cx + s, cy), (cx, cy + s)],
        style="F",
    )
    pdf.ln(6)

    pdf.set_font(FUENTE, "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, OBRA_LUGAR, ln=True, align="C")
    pdf.cell(0, 6, OBRA_FECHA, ln=True, align="C")
    pdf.cell(0, 6, OBRA_GRUPO, ln=True, align="C")
    pdf.ln(6)

    # --- RECUADRO DE DATOS ---
    y_rect = pdf.get_y()
    h_rect = 105

    pdf.set_fill_color(220, 220, 220)
    pdf.rect(17, y_rect + 1.5, 176, h_rect, style="F")

    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(*dorado)
    pdf.set_line_width(0.6)
    pdf.rect(15, y_rect, 176, h_rect, style="DF")

    pdf.set_fill_color(*dorado)
    pdf.rect(15, y_rect, 176, 10, style="F")
    pdf.set_xy(15, y_rect + 1.5)
    pdf.set_font(FUENTE, "B", 12)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(176, 7, "D A T O S   D E L   B O L E T O", ln=True, align="C")

    def _linea(label, valor, y_pos, label_w=32):
        pdf.set_xy(22, y_pos)
        pdf.set_font(FUENTE, "B", 10)
        pdf.set_text_color(*marron)
        pdf.cell(label_w, 8, f"{label}:", ln=False)
        pdf.set_font(FUENTE, "", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(85, 8, str(valor), ln=True)

    y0 = y_rect + 16
    _linea("Folio", str(int(datos_venta.get("Folio", 0))), y0)
    _linea("Nombre", str(datos_venta.get("Nombre", "")), y0 + 11)
    _linea("Modalidad", str(datos_venta.get("Circunstancia", "")), y0 + 22)
    _linea("Costo", precio_fmt(datos_venta.get("Costo del boleto", 0)) + " MXN", y0 + 33)
    _linea("Modo de pago", str(datos_venta.get("Modo de pago", "")), y0 + 44)
    _linea("Vendido por", str(datos_venta.get("Vendido por:", "")), y0 + 55)

    # --- QR ---
    if QR_DISPONIBLE:
        try:
            qr_data = (
                f"THAAY|FOLIO:{int(datos_venta.get('Folio', 0))}"
                f"|NOMBRE:{datos_venta.get('Nombre', '')}"
                f"|FECHA:{datos_venta.get('Fecha', '')}"
            )
            qr = qrcode.QRCode(version=1, box_size=8, border=1)
            qr.add_data(qr_data)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")

            qr_buffer = io.BytesIO()
            qr_img.save(qr_buffer, format="PNG")
            qr_buffer.seek(0)

            pdf.image(qr_buffer, x=142, y=y0, w=42)
            qr_buffer.close()

            pdf.set_xy(140, y0 + 44)
            pdf.set_font(FUENTE, "I", 8)
            pdf.set_text_color(*gris)
            pdf.cell(46, 4, "Escanea para validar", ln=True, align="C")
            pdf.set_x(140)
            pdf.cell(46, 4, "tu boleto en la entrada", ln=True, align="C")
        except Exception:
            pass

    pdf.set_xy(15, y_rect + h_rect - 12)
    pdf.set_font(FUENTE, "I", 9)
    pdf.set_text_color(*gris)
    pdf.cell(
        176, 5,
        f"Fecha de emisión: {datos_venta.get('Fecha', '')}  ·  "
        f"Folio único: {int(datos_venta.get('Folio', 0))}",
        ln=True, align="C",
    )

    pdf.set_y(y_rect + h_rect + 10)

    # --- TALÓN DESPRENDIBLE ---
    pdf.set_draw_color(*dorado_suave)
    pdf.set_line_width(0.4)
    try:
        pdf.set_dash_pattern(dash=1.5, gap=1.5)
        y_perf = pdf.get_y()
        pdf.line(15, y_perf, 195, y_perf)
        pdf.set_dash_pattern()
    except Exception:
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)

    pdf.set_font(FUENTE, "B", 11)
    pdf.set_text_color(*marron)
    pdf.cell(0, 6, "Información importante", ln=True, align="C")
    pdf.ln(2)
    pdf.set_font(FUENTE, "", 10)
    pdf.set_text_color(*gris)
    pdf.multi_cell(
        0, 5,
        "· Este boleto es personal e intransferible.\n"
        "· El asiento es libre: puedes sentarte donde prefieras. El folio es solo control interno.\n"
        "· Preséntate en la entrada 15 minutos antes de la función.\n"
        "· Conserva este boleto durante toda la función.",
        align="C",
    )
    pdf.ln(6)

    # --- PIE ---
    pdf.set_draw_color(*dorado)
    pdf.set_line_width(0.4)
    y_pie = pdf.get_y()
    pdf.line(70, y_pie, 140, y_pie)
    pdf.ln(4)

    pdf.set_font(FUENTE, "B", 12)
    pdf.set_text_color(*dorado)
    pdf.cell(0, 7, "COMPAÑÍA DE TEATRO THAAY LERMA", ln=True, align="C")
    pdf.set_font(FUENTE, "I", 9)
    pdf.set_text_color(*gris)
    pdf.cell(0, 5, "Gracias por apoyar el teatro local · ¡Disfruta la función!", ln=True, align="C")

    salida = pdf.output()
    return bytes(salida) if isinstance(salida, (bytearray, memoryview)) else salida


# ============================================================
# PERSISTENCIA
# ============================================================
def _config_default():
    return {"circunstancia_actual": "Entrada General", "next_folio": 1}


def cargar_config():
    cfg = _config_default()
    if ARCHIVO_CONFIG.exists():
        try:
            with open(ARCHIVO_CONFIG, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in cfg.items():
                    data.setdefault(k, v)
                if data["circunstancia_actual"] not in MODALIDADES:
                    data["circunstancia_actual"] = "Entrada General"
                data["next_folio"] = max(1, int(data.get("next_folio", 1)))
                return data
        except Exception:
            pass
    return cfg


def guardar_config(cfg):
    with open(ARCHIVO_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def migrar_csv_viejo(df):
    if df.empty:
        return df
    needs = False
    for _, row in df.iterrows():
        if "," in str(row.get("Folio", "")):
            needs = True
            break
        try:
            if int(float(str(row.get("Cantidad de boletos", 1)))) > 1:
                needs = True
                break
        except (ValueError, TypeError):
            pass
    if not needs:
        return df

    nuevos = []
    for _, row in df.iterrows():
        f_raw = str(row.get("Folio", "")).strip()
        try:
            cant = int(float(str(row.get("Cantidad de boletos", 1))))
        except (ValueError, TypeError):
            cant = 1

        folios = []
        if "," in f_raw:
            try:
                folios = sorted(int(x.strip()) for x in f_raw.split(",") if x.strip())
            except ValueError:
                folios = []
        else:
            try:
                start = int(float(f_raw))
                folios = list(range(start, start + cant))
            except ValueError:
                folios = []

        for folio in folios:
            nuevo = row.to_dict()
            nuevo["Folio"] = folio
            nuevo["Cantidad de boletos"] = 1
            nuevos.append(nuevo)

    return pd.DataFrame(nuevos, columns=df.columns) if nuevos else df


def cargar_ventas():
    if not ARCHIVO_VENTAS.exists():
        return pd.DataFrame(columns=COLUMNAS)
    try:
        df = pd.read_csv(ARCHIVO_VENTAS, keep_default_na=False,
                         dtype=str, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=COLUMNAS)

    for col in COLUMNAS:
        if col not in df.columns:
            if col == "Cantidad de boletos":
                df[col] = "1"
            elif col == "Boleto digital":
                df[col] = "No"
            elif col == "Estado":
                df[col] = "Válido"
            else:
                df[col] = ""
    df = df[COLUMNAS].copy()

    df = migrar_csv_viejo(df)

    df["Folio"] = pd.to_numeric(df["Folio"], errors="coerce").fillna(0).astype(int)
    df["Cantidad de boletos"] = pd.to_numeric(df["Cantidad de boletos"],
                                               errors="coerce").fillna(1).astype(int)
    df["Costo del boleto"] = pd.to_numeric(df["Costo del boleto"],
                                            errors="coerce").fillna(0.0).astype(float)
    for col in ["Nombre", "Modo de pago", "Vendido por:", "Fecha",
                "Teléfono de contacto", "e-mail", "Boleto digital",
                "Circunstancia", "Comentarios", "Estado", "Check-in"]:
        df[col] = df[col].astype(str).fillna("")

    df["Boleto digital"] = df["Boleto digital"].apply(
        lambda v: "Sí" if _es_si(v) else "No"
    )
    # Normalizar Estado (por si viene vacío o "nan")
    df["Estado"] = df["Estado"].replace({"": "Válido", "nan": "Válido", "None": "Válido"})
    df["Estado"] = df["Estado"].where(df["Estado"].isin(ESTADOS_BOLETO), "Válido")

    return df.reset_index(drop=True)


def guardar_ventas(df):
    df.to_csv(ARCHIVO_VENTAS, index=False, encoding="utf-8-sig")


# ============================================================
# VALIDACIÓN DE DUPLICADOS
# ============================================================
def buscar_coincidencias(df, nombre, fecha, limite=5):
    if df.empty:
        return []
    nombre_norm = normalizar(nombre)
    grupos = {}
    for _, row in df.iterrows():
        ratio = SequenceMatcher(None, nombre_norm, normalizar(row["Nombre"])).ratio()
        if ratio >= UMBRAL_SIMILITUD:
            key = (normalizar(row["Nombre"]), str(row["Fecha"]))
            if key not in grupos:
                grupos[key] = {
                    "Folios": [],
                    "Nombre": row["Nombre"],
                    "Fecha": str(row["Fecha"]),
                    "Vendido por:": row["Vendido por:"],
                    "_ratio": ratio,
                    "_misma_fecha": str(row["Fecha"]) == str(fecha),
                }
            grupos[key]["Folios"].append(int(row["Folio"]))
            grupos[key]["_ratio"] = max(grupos[key]["_ratio"], ratio)

    out = []
    for g in grupos.values():
        out.append({
            "Folio(s)": ", ".join(str(f) for f in sorted(g["Folios"])),
            "Nombre": g["Nombre"],
            "Fecha": g["Fecha"],
            "Boletos": len(g["Folios"]),
            "Similitud": f"{g['_ratio'] * 100:.0f}%",
            "Vendido por:": g["Vendido por:"],
            "_ratio": g["_ratio"],
            "_misma_fecha": g["_misma_fecha"],
        })
    out.sort(key=lambda x: x["_ratio"], reverse=True)
    return out[:limite]


# ============================================================
# CONFIGURACIÓN DE PÁGINA Y CARGA INICIAL
# ============================================================
st.set_page_config(
    page_title=f"Boletos · {OBRA_TITULO}",
    page_icon="🎭",
    layout="wide",
)


# ============================================================
# AUTENTICACIÓN CON CONTRASEÑA
# ============================================================
def check_password():
    """Devuelve True si el usuario ingresó la contraseña correcta."""
    def password_entered():
        try:
            password_correcta = st.secrets.get("APP_PASSWORD", "")
        except Exception:
            password_correcta = ""
        if st.session_state.get("password", "") == password_correcta:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.markdown("## 🔒 Acceso restringido")
    st.caption(
        f"Panel interno del Taller de Teatro · {OBRA_TITULO}  \n"
        "Solo integrantes del grupo. Ingresa la contraseña para continuar."
    )
    st.text_input(
        "Contraseña",
        type="password",
        on_change=password_entered,
        key="password",
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Contraseña incorrecta. Intenta de nuevo.")
    return False


if not check_password():
    st.stop()


cfg = cargar_config()
df = cargar_ventas()
recalcular_next_folio(df, cfg)

if "pending_sale" not in st.session_state:
    st.session_state.pending_sale = None
if "coincidencias" not in st.session_state:
    st.session_state.coincidencias = []
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None
if "ultima_venta_pdf" not in st.session_state:
    st.session_state.ultima_venta_pdf = None

boletos_vendidos = len(df)
disponibles = TOTAL_ASIENTOS - boletos_vendidos
recaudado = float(df["Costo del boleto"].sum()) if not df.empty else 0.0


# ============================================================
# CABECERA
# ============================================================
col_logo, col_info = st.columns([1, 4], vertical_alignment="center")

with col_logo:
    if ARCHIVO_LOGO.exists():
        st.image(str(ARCHIVO_LOGO), width=200)
    else:
        st.markdown("<div style='font-size:80px;text-align:center;'>🎭</div>",
                    unsafe_allow_html=True)

with col_info:
    st.markdown(
        "<div style='background:#F5F0E8;padding:6px 12px;border-radius:6px;"
        "display:inline-block;font-size:12px;color:#7B5E1A;font-weight:600;"
        "letter-spacing:1px;margin-bottom:8px;'>"
        "🔒 PANEL INTERNO · TALLER DE TEATRO</div>",
        unsafe_allow_html=True,
    )
    st.title(f"🎭 {OBRA_TITULO}")
    st.markdown(f"**{OBRA_AUTOR}**")
    st.caption(f"📍 {OBRA_LUGAR}")
    st.caption(f"📅 {OBRA_FECHA}  ·  👥 {OBRA_GRUPO}")
    st.caption(
        f"🎟️ Aforo: **{TOTAL_ASIENTOS} asientos** · Entrada general · "
        f"El público puede sentarse donde quiera; el folio es solo control interno."
    )

st.markdown(
    f"<div style='background:#F5F0E8;border-left:4px solid #B8860B;"
    f"padding:8px 14px;border-radius:4px;margin-top:10px;font-size:13px;color:#3E2723;'>"
    f"💳 <b>Transferencia SPEI</b> · Entidad financiera: <b>{SPEI_BANCO}</b> · "
    f"Beneficiario: <b>{SPEI_BENEFICIARIO}</b> · "
    f"CLABE: <b>{SPEI_CLABE}</b><br>"
    f"Concepto sugerido: <b>'{SPEI_CONCEPTO}'</b>"
    f"</div>",
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("💵 Precios oficiales")
    for nombre_mod, info in MODALIDADES.items():
        cant = info["cantidad"]
        precio = info["precio"]
        if cant > 1:
            st.markdown(f"**{nombre_mod}**  \n**Para fecha posterior a la Preventa**")
        else:
            st.markdown(f"**{nombre_mod}** · {precio_fmt(precio)}")

    st.divider()
    st.header("📊 Resumen")
    c1, c2 = st.columns(2)
    c1.metric("Vendidos", boletos_vendidos)
    c2.metric("Disponibles", disponibles)
    st.metric("Recaudado", precio_fmt(recaudado))

    if cfg["next_folio"] > TOTAL_ASIENTOS and disponibles > 0:
        st.info(
            f"ℹ️ Folios consecutivos agotados. Las próximas ventas reutilizarán "
            f"folios liberados (hay **{disponibles}** disponibles)."
        )

    st.divider()
    st.header("↩️ Deshacer último folio")
    if df.empty:
        st.caption("No hay ventas registradas.")
    else:
        ultima = df.iloc[-1]
        st.warning(
            f"**Folio:** {int(ultima['Folio'])}  \n"
            f"**{ultima['Nombre']}**  \n"
            f"**{precio_fmt(ultima['Costo del boleto'])}**"
        )
        st.caption(
            "Elimina solo el último folio registrado. Si una venta fue de varios "
            "boletos, hazlo varias veces."
        )
        confirmar = st.checkbox("Confirmo eliminar este folio", key="chk_undo")
        if st.button("🗑️ Deshacer", disabled=not confirmar, use_container_width=True):
            folio_undo = int(ultima["Folio"])
            df = df.iloc[:-1].reset_index(drop=True)
            guardar_ventas(df)
            recalcular_next_folio(df, cfg)
            st.toast(f"↩️ Folio {folio_undo} eliminado.")
            st.rerun()


# ============================================================
# TABS
# ============================================================
tab_reg, tab_ver, tab_validar, tab_graf, tab_caja, tab_edit, tab_exp = st.tabs(
    ["➕ Registrar", "📋 Ver ventas", "🎫 Validar entrada",
     "📈 Gráficas", "💰 Corte de Caja", "✏️ Editar", "📤 Exportar"]
)


# ============================================================
# TAB 1: REGISTRAR
# ============================================================
with tab_reg:
    st.subheader("➕ Registrar nueva venta")

    if disponibles <= 0:
        st.error(f"🚫 Aforo completo: los {TOTAL_ASIENTOS} asientos están vendidos.")
    else:
        if "reg_modalidad" not in st.session_state:
            st.session_state.reg_modalidad = cfg["circunstancia_actual"]
        if "reg_costo" not in st.session_state:
            st.session_state.reg_costo = float(MODALIDADES[st.session_state.reg_modalidad]["precio"])
        if "reg_cantidad" not in st.session_state:
            st.session_state.reg_cantidad = int(MODALIDADES[st.session_state.reg_modalidad]["cantidad"])

        if st.session_state.reg_cantidad > disponibles:
            st.session_state.reg_cantidad = disponibles

        def _reset_defaults():
            mod = st.session_state.get("reg_modalidad", "Entrada General")
            st.session_state.reg_costo = float(MODALIDADES[mod]["precio"])
            st.session_state.reg_cantidad = min(int(MODALIDADES[mod]["cantidad"]), disponibles)

        modalidad_sel = st.radio(
            "Modalidad de venta",
            options=list(MODALIDADES.keys()),
            key="reg_modalidad",
            on_change=_reset_defaults,
            horizontal=True,
        )
        st.caption(f"ℹ️ {MODALIDADES[modalidad_sel]['nota']}")

        col1, col2 = st.columns(2)
        with col1:
            nombre = st.text_input("Nombre del comprador *", key="reg_nombre",
                                   placeholder="Nombre completo")
            costo = st.number_input(
                "Costo unitario por boleto ($) *",
                min_value=0.0, step=5.0, key="reg_costo",
                help="Precio por boleto, NO el total de la venta.",
            )
            cantidad = st.number_input(
                "Cantidad de boletos *",
                min_value=1, max_value=max(1, disponibles), step=1,
                key="reg_cantidad",
            )
            modo_pago = st.selectbox("Modo de pago *", MODOS_PAGO, key="reg_pago")
        with col2:
            vendedor = st.selectbox("Vendido por: *", VENDEDORES, key="reg_vend")
            fecha_venta = st.date_input("Fecha de venta *", value=date.today(), key="reg_fecha")
            telefono = st.text_input("Teléfono de contacto (opcional)", key="reg_tel",
                                     placeholder="Ej: 55 1234 5678")
            email = st.text_input("Correo electrónico (opcional)", key="reg_email",
                                  placeholder="Ej: cliente@correo.com")

        comentarios = st.text_area(
            "Comentarios (opcional)", key="reg_com", height=80,
            placeholder="Ej: pagó con $200, se dio cambio · familiar de Sara · recoge el día del evento",
        )

        entrega_pdf = st.checkbox(
            "🎟️ El cliente solicita boleto digital (PDF con QR)",
            key="reg_pdf",
            help="Marca esta casilla solo si el cliente pidió recibir su boleto "
                 "en formato digital (por correo, WhatsApp, etc.).",
        )

        folios_preview = asignar_folios(df, int(cantidad), cfg["next_folio"])
        total_venta = float(costo) * int(cantidad)
        if folios_preview is None:
            st.error(f"🚫 No hay suficientes folios disponibles para asignar {int(cantidad)} boleto(s).")
        else:
            st.info(
                f"📌 **Resumen de la venta**  \n"
                f"Folio(s): **{folios_compact(folios_preview)}**  \n"
                f"Cantidad: **{int(cantidad)}**  \n"
                f"Costo unitario: **\\${costo:.2f}**  \n"
                f"Total: **\\${total_venta:.2f}**  \n"
                f"Boleto digital: **{'Sí' if entrega_pdf else 'No'}**"
            )

        registrar_disabled = folios_preview is None
        if st.button("✅ Registrar venta", type="primary",
                     use_container_width=True, disabled=registrar_disabled):
            if not nombre.strip():
                st.error("El nombre del comprador es obligatorio.")
            else:
                folios_finales = asignar_folios(df, int(cantidad), cfg["next_folio"])
                if folios_finales is None:
                    st.error("Ya no hay folios disponibles para esta venta.")
                else:
                    base = {
                        "Nombre": nombre.strip(),
                        "Costo del boleto": float(costo),
                        "Cantidad de boletos": 1,
                        "Modo de pago": modo_pago,
                        "Vendido por:": vendedor,
                        "Fecha": fecha_venta.strftime("%Y-%m-%d"),
                        "Teléfono de contacto": telefono.strip(),
                        "e-mail": email.strip(),
                        "Boleto digital": "Sí" if entrega_pdf else "No",
                        "Circunstancia": modalidad_sel,
                        "Comentarios": comentarios.strip(),
                        "Estado": "Válido",
                        "Check-in": "",
                    }
                    nuevas_filas = [{"Folio": f, **base} for f in folios_finales]

                    coinc = buscar_coincidencias(df, nombre.strip(), base["Fecha"])
                    if coinc:
                        st.session_state.pending_sale = nuevas_filas
                        st.session_state.coincidencias = coinc
                        st.rerun()
                    else:
                        df = pd.concat(
                            [df, pd.DataFrame(nuevas_filas, columns=COLUMNAS)],
                            ignore_index=True,
                        )
                        guardar_ventas(df)
                        recalcular_next_folio(df, cfg)
                        cfg["circunstancia_actual"] = modalidad_sel
                        guardar_config(cfg)

                        if entrega_pdf:
                            st.session_state.ultima_venta_pdf = nuevas_filas

                        for k in ["reg_nombre", "reg_tel", "reg_email", "reg_com", "reg_pdf"]:
                            st.session_state.pop(k, None)
                        st.toast(
                            f"✅ Venta registrada · Folio(s) "
                            f"{folios_compact(folios_finales)} · {precio_fmt(total_venta)}"
                        )
                        st.rerun()

        # --- Descarga PDFs ---
        if st.session_state.ultima_venta_pdf:
            st.divider()
            st.success("✅ Venta registrada · Puedes descargar el boleto digital.")
            for i, boleto in enumerate(st.session_state.ultima_venta_pdf):
                try:
                    pdf_bytes = generar_boleto_pdf(boleto)
                    st.download_button(
                        label=f"📥 Descargar Boleto · Folio {int(boleto['Folio'])}",
                        data=pdf_bytes,
                        file_name=(
                            f"boleto_folio_{int(boleto['Folio'])}_"
                            f"{boleto['Nombre'].replace(' ', '_')}.pdf"
                        ),
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"descarga_pdf_{i}",
                    )
                except Exception as e:
                    st.error(f"Error generando PDF del folio {boleto['Folio']}: {e}")

            if st.button("🔄 Cerrar y continuar", use_container_width=True, key="cerrar_pdf"):
                st.session_state.ultima_venta_pdf = None
                st.rerun()

        # ---------- DIÁLOGO DE DUPLICADOS ----------
        if st.session_state.pending_sale:
            st.divider()
            p_filas = st.session_state.pending_sale
            coinc = st.session_state.coincidencias
            hay_misma = any(c["_misma_fecha"] for c in coinc)

            if hay_misma:
                st.error(
                    f"🚨 **Posible duplicado.** {len(coinc)} coincidencia(s) con nombre similar, "
                    f"y **al menos una coincide también en fecha**."
                )
            else:
                st.warning(
                    f"ℹ️ **Mismo nombre, distinta fecha.** {len(coinc)} coincidencia(s) con nombre similar."
                )

            p0 = p_filas[0]
            st.markdown("**Venta pendiente de confirmar:**")
            st.dataframe(
                pd.DataFrame([{
                    "Folio(s)": ", ".join(str(f["Folio"]) for f in p_filas),
                    "Nombre": p0["Nombre"],
                    "Fecha": p0["Fecha"],
                    "Cantidad": len(p_filas),
                    "Costo unit.": p0["Costo del boleto"],
                    "Total": p0["Costo del boleto"] * len(p_filas),
                    "Boleto digital": p0["Boleto digital"],
                    "Vendido por:": p0["Vendido por:"],
                }]),
                hide_index=True, use_container_width=True,
            )

            st.markdown("**Coincidencias existentes:**")
            st.dataframe(
                pd.DataFrame([{k: v for k, v in c.items() if not k.startswith("_")} for c in coinc]),
                hide_index=True, use_container_width=True,
            )

            ca, cb = st.columns(2)
            if ca.button("✅ Confirmar y registrar", type="primary", use_container_width=True):
                df = pd.concat(
                    [df, pd.DataFrame(st.session_state.pending_sale, columns=COLUMNAS)],
                    ignore_index=True,
                )
                guardar_ventas(df)
                recalcular_next_folio(df, cfg)
                cfg["circunstancia_actual"] = p0["Circunstancia"]
                guardar_config(cfg)

                if p0.get("Boleto digital") == "Sí":
                    st.session_state.ultima_venta_pdf = st.session_state.pending_sale

                st.session_state.pending_sale = None
                st.session_state.coincidencias = []
                for k in ["reg_nombre", "reg_tel", "reg_email", "reg_com", "reg_pdf"]:
                    st.session_state.pop(k, None)
                st.toast("✅ Venta registrada.")
                st.rerun()
            if cb.button("❌ Cancelar", use_container_width=True):
                st.session_state.pending_sale = None
                st.session_state.coincidencias = []
                st.toast("Registro cancelado.")
                st.rerun()


# ============================================================
# TAB 2: VER VENTAS
# ============================================================
with tab_ver:
    st.subheader("📋 Ventas registradas")

    if df.empty:
        st.info("Aún no hay ventas registradas.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            f_vend = st.multiselect("Vendedor", VENDEDORES, default=VENDEDORES)
        with c2:
            f_pago = st.multiselect("Modo de pago", MODOS_PAGO, default=MODOS_PAGO)
        with c3:
            f_circ = st.multiselect("Modalidad", list(MODALIDADES.keys()),
                                    default=list(MODALIDADES.keys()))
        with c4:
            buscar = st.text_input("Buscar (nombre, folio, teléfono, e-mail)")

        c5, _c6 = st.columns([1, 3])
        with c5:
            f_estado = st.multiselect("Estado del boleto", ESTADOS_BOLETO,
                                      default=ESTADOS_BOLETO)

        df_f = df.copy()
        if f_vend:
            df_f = df_f[df_f["Vendido por:"].isin(f_vend)]
        if f_pago:
            df_f = df_f[df_f["Modo de pago"].isin(f_pago)]
        if f_circ:
            df_f = df_f[df_f["Circunstancia"].isin(f_circ)]
        if f_estado:
            df_f = df_f[df_f["Estado"].isin(f_estado)]
        if buscar:
            q = buscar.strip().lower()
            mask = (
                df_f["Nombre"].str.lower().str.contains(q, na=False) |
                df_f["Folio"].astype(str).str.contains(q, na=False) |
                df_f["Teléfono de contacto"].astype(str).str.lower().str.contains(q, na=False) |
                df_f["e-mail"].astype(str).str.lower().str.contains(q, na=False)
            )
            df_f = df_f[mask]

        df_show = df_f.copy()
        df_show["Total"] = (df_show["Costo del boleto"] * df_show["Cantidad de boletos"]).round(2)
        df_show = df_show[[
            "Folio", "Nombre", "Costo del boleto", "Cantidad de boletos", "Total",
            "Modo de pago", "Vendido por:", "Fecha", "Teléfono de contacto",
            "e-mail", "Boleto digital", "Circunstancia", "Comentarios",
            "Estado", "Check-in",
        ]].rename(columns={"Costo del boleto": "Costo unit."})

        st.caption(f"Mostrando **{len(df_show)}** boleto(s) vendido(s).")
        st.dataframe(
            df_show, hide_index=True, use_container_width=True,
            column_config={
                "Folio": st.column_config.NumberColumn("Folio", format="%d"),
                "Costo unit.": st.column_config.NumberColumn("Costo unit.", format="$%.2f"),
                "Total": st.column_config.NumberColumn("Total", format="$%.2f"),
                "Cantidad de boletos": st.column_config.NumberColumn("Cant.", format="%d"),
            },
        )

        st.divider()
        st.subheader("Resumen por vendedor")
        df_res = df.copy()
        df_res["Total"] = df_res["Costo del boleto"]
        resumen = df_res.groupby("Vendido por:", as_index=False).agg(
            Boletos=("Folio", "count"),
            Recaudado=("Total", "sum"),
        )
        resumen["Recaudado"] = resumen["Recaudado"].round(2)
        st.dataframe(
            resumen, hide_index=True, use_container_width=True,
            column_config={
                "Recaudado": st.column_config.NumberColumn("Recaudado", format="$%.2f"),
            },
        )

        st.subheader("Resumen por modalidad")
        resumen_mod = df_res.groupby("Circunstancia", as_index=False).agg(
            Boletos=("Folio", "count"),
            Recaudado=("Total", "sum"),
        )
        resumen_mod["Recaudado"] = resumen_mod["Recaudado"].round(2)
        st.dataframe(
            resumen_mod, hide_index=True, use_container_width=True,
            column_config={
                "Recaudado": st.column_config.NumberColumn("Recaudado", format="$%.2f"),
            },
        )


# ============================================================
# TAB 3: VALIDAR ENTRADA
# ============================================================
with tab_validar:
    st.subheader("🎫 Validación de boletos en la entrada")
    st.caption(
        "Escanea el QR del boleto con la cámara web, o ingresa el folio manualmente. "
        "El sistema marca el boleto como 'Usado' para evitar duplicados."
    )

    # Aviso del estado de las librerías
    avisos = []
    if not QR_DISPONIBLE:
        avisos.append("No se generan QRs nuevos (falta `qrcode`).")
    if not PYZBAR_DISPONIBLE and not CV_DISPONIBLE:
        avisos.append("No hay lector de QR disponible. Instala `pyzbar` o `opencv-python-headless`.")
    elif not PYZBAR_DISPONIBLE:
        avisos.append("Usando OpenCV como lector (pyzbar no está disponible).")
    if avisos:
        st.info(" · ".join(avisos))

    if df.empty:
        st.info("Aún no hay boletos vendidos.")
    else:
        # --- Métricas rápidas ---
        total_validos = int((df["Estado"] == "Válido").sum())
        total_usados = int((df["Estado"] == "Usado").sum())
        total_cancelados = int((df["Estado"] == "Cancelado").sum())

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("🎟️ Total boletos", len(df))
        mc2.metric("🟢 Válidos (sin usar)", total_validos)
        mc3.metric("✅ Usados (ya entraron)", total_usados)
        mc4.metric("🚫 Cancelados", total_cancelados)

        st.divider()

        # --- Selector de método ---
        modo_escaneo = st.radio(
            "Método de validación",
            options=["📷 Cámara", "⌨️ Folio manual"],
            horizontal=True,
            key="modo_escaneo",
        )

        folio_a_validar = None

        if modo_escaneo == "📷 Cámara":
            if not (PYZBAR_DISPONIBLE or CV_DISPONIBLE):
                st.error(
                    "No hay lector de QR disponible. Instala con:\n"
                    "`pip install opencv-python-headless` o `pip install pyzbar`"
                )
            else:
                foto = st.camera_input("Apunta la cámara al código QR del boleto",
                                        key="camara_qr")
                if foto is not None:
                    with st.spinner("Leyendo QR..."):
                        texto_qr = leer_qr_desde_bytes(foto.getvalue())
                    if texto_qr:
                        st.success(f"QR leído correctamente.")
                        with st.expander("Ver contenido del QR"):
                            st.code(texto_qr, language="text")
                        folio_a_validar = extraer_folio_de_qr(texto_qr)
                        if folio_a_validar is None:
                            st.warning(
                                "El QR no contiene un folio reconocible. "
                                "Probablemente no fue generado por este sistema."
                            )
                    else:
                        st.warning(
                            "No se detectó ningún QR en la imagen. Intenta con mejor luz, "
                            "más cerca, o sin reflejos."
                        )

        else:  # Folio manual
            folio_input = st.text_input(
                "Ingresa el folio del boleto",
                placeholder="Ej: 42",
                key="folio_manual",
            )
            if folio_input.strip():
                try:
                    folio_a_validar = int(folio_input.strip())
                except ValueError:
                    st.warning("El folio debe ser un número entero.")

        # --- Resultado de la validación ---
        if folio_a_validar is not None:
            resultado = validar_folio(df, folio_a_validar)
            st.divider()

            if not resultado["existe"]:
                st.error(
                    f"❌ **Boleto no encontrado.** El folio **{folio_a_validar}** "
                    f"no existe en el registro de ventas. Probablemente es un boleto falso."
                )
            else:
                # Datos del boleto en tarjeta
                with st.container(border=True):
                    ca, cb, cc = st.columns([1, 2, 1])
                    with ca:
                        st.markdown(f"### Folio **{folio_a_validar}**")
                    with cb:
                        st.markdown(f"**{resultado['nombre']}**")
                        st.caption(
                            f"{resultado['modalidad']} · "
                            f"Vendido por: {resultado['vendido_por']}"
                        )
                    with cc:
                        estado_actual = resultado["estado"]
                        if estado_actual == "Válido":
                            st.success("🟢 Válido")
                        elif estado_actual == "Usado":
                            st.warning("🟡 Usado")
                        else:
                            st.error("🔴 Cancelado")

                estado = resultado["estado"]

                if estado == "Usado":
                    st.warning(
                        f"⚠️ **Boleto ya utilizado.**  \n"
                        f"Este folio fue escaneado el **{resultado['check_in']}**.  \n"
                        f"No permitas el acceso con este boleto."
                    )
                elif estado == "Cancelado":
                    st.error(
                        "🚫 **Boleto cancelado.** Este folio fue anulado y no es válido "
                        "para acceso."
                    )
                else:  # Válido
                    st.success("✅ **Boleto válido. Puede pasar.**")
                    if st.button(
                        "🎟️ Marcar como usado y permitir acceso",
                        type="primary",
                        use_container_width=True,
                        key=f"marcar_usado_{folio_a_validar}",
                    ):
                        df = marcar_boleto_usado(df, folio_a_validar)
                        st.toast(f"✅ Folio {folio_a_validar} validado. ¡Buen espectáculo!")
                        st.rerun()

        st.divider()

        # --- Acciones administrativas ---
        with st.expander("🔧 Acciones administrativas (usar con cuidado)"):
            st.caption(
                "Solo para el responsable del evento: reactivar o cancelar boletos "
                "cuando haya un error humano, devolución, etc."
            )
            c1, c2 = st.columns(2)
            with c1:
                folio_reactivar = st.number_input(
                    "Folio a reactivar (marcar como Válido)",
                    min_value=0, step=1, key="folio_reactivar",
                )
                if st.button("♻️ Reactivar", use_container_width=True, key="btn_reactivar"):
                    idxs = df.index[df["Folio"].astype(int) == int(folio_reactivar)].tolist()
                    if idxs:
                        df = cambiar_estado_boleto(df, folio_reactivar, "Válido")
                        st.toast(f"♻️ Folio {folio_reactivar} reactivado.")
                        st.rerun()
                    else:
                        st.warning("Folio no encontrado.")

            with c2:
                folio_cancelar = st.number_input(
                    "Folio a cancelar",
                    min_value=0, step=1, key="folio_cancelar",
                )
                if st.button("🚫 Cancelar boleto", use_container_width=True, key="btn_cancelar"):
                    idxs = df.index[df["Folio"].astype(int) == int(folio_cancelar)].tolist()
                    if idxs:
                        df = cambiar_estado_boleto(df, folio_cancelar, "Cancelado")
                        st.toast(f"🚫 Folio {folio_cancelar} cancelado.")
                        st.rerun()
                    else:
                        st.warning("Folio no encontrado.")

        # --- Historial reciente de accesos ---
        st.divider()
        st.subheader("⏱️ Últimos accesos registrados")
        df_usados = df[df["Estado"] == "Usado"].copy()
        if df_usados.empty:
            st.caption("Aún no se ha validado ningún boleto.")
        else:
            df_usados = df_usados.sort_values("Check-in", ascending=False).head(20)
            st.dataframe(
                df_usados[["Folio", "Nombre", "Check-in", "Vendido por:", "Circunstancia"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Folio": st.column_config.NumberColumn("Folio", format="%d"),
                },
            )


# ============================================================
# TAB 4: GRÁFICAS
# ============================================================
with tab_graf:
    st.subheader("📈 Análisis de ventas")

    if df.empty:
        st.info("Aún no hay datos para graficar.")
    else:
        df_g = df.copy()
        df_g["Total"] = df_g["Costo del boleto"]
        df_g["Fecha_dt"] = pd.to_datetime(df_g["Fecha"], errors="coerce")
        df_g["Vendedor"] = df_g["Vendido por:"]

        pct = boletos_vendidos / TOTAL_ASIENTOS
        st.markdown(f"### Progreso · {boletos_vendidos} / {TOTAL_ASIENTOS} asientos")
        st.progress(min(pct, 1.0), text=f"{pct * 100:.1f}% vendido")
        st.markdown("")

        df_valid = df_g.dropna(subset=["Fecha_dt"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📅 Boletos vendidos por día")
            if not df_valid.empty:
                por_dia = (
                    df_valid.groupby(df_valid["Fecha_dt"].dt.date)
                    .size().reset_index(name="Boletos")
                )
                por_dia.columns = ["Fecha", "Boletos"]
                st.bar_chart(por_dia, x="Fecha", y="Boletos", use_container_width=True)
            else:
                st.caption("Sin datos con fecha válida.")
        with c2:
            st.markdown("#### 💰 Recaudación por día ($)")
            if not df_valid.empty:
                rec_dia = (
                    df_valid.groupby(df_valid["Fecha_dt"].dt.date)["Total"]
                    .sum().reset_index()
                )
                rec_dia.columns = ["Fecha", "Recaudado"]
                st.bar_chart(rec_dia, x="Fecha", y="Recaudado", use_container_width=True)
            else:
                st.caption("Sin datos con fecha válida.")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### 👤 Boletos por vendedor")
            por_vend = df_g.groupby("Vendedor").size().reset_index(name="Boletos")
            st.bar_chart(por_vend, x="Vendedor", y="Boletos", use_container_width=True)
        with c4:
            st.markdown("#### 💵 Recaudación por vendedor ($)")
            rec_vend = df_g.groupby("Vendedor")["Total"].sum().reset_index()
            rec_vend.columns = ["Vendedor", "Recaudado"]
            st.bar_chart(rec_vend, x="Vendedor", y="Recaudado", use_container_width=True)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Recaudación total", precio_fmt(df_g["Total"].sum()))
        m2.metric("Ticket promedio", precio_fmt(df_g["Total"].mean()))
        m3.metric("Boletos vendidos", len(df_g))


# ============================================================
# TAB 5: CORTE DE CAJA
# ============================================================
with tab_caja:
    st.subheader("💰 Corte de Caja")
    st.caption("Resumen diario de ventas para rendición de cuentas y control interno.")

    if df.empty:
        st.info("Aún no hay ventas registradas.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            fecha_corte = st.date_input(
                "Fecha del corte",
                value=date.today(),
                key="fecha_corte",
                help="Selecciona el día del cual quieres el resumen.",
            )
        with c2:
            modo_vista = st.radio(
                "Vista",
                options=["Un día", "Rango"],
                horizontal=True,
                key="modo_corte",
            )
        with c3:
            if modo_vista == "Rango":
                rango = st.date_input(
                    "Rango de fechas",
                    value=(fecha_corte - timedelta(days=6), fecha_corte),
                    key="rango_corte",
                )
            else:
                rango = None

        df_corte = df.copy()
        df_corte["Fecha_dt"] = pd.to_datetime(df_corte["Fecha"], errors="coerce")

        if modo_vista == "Un día":
            fecha_str = fecha_corte.strftime("%Y-%m-%d")
            df_corte = df_corte[df_corte["Fecha"] == fecha_str]
            titulo_rango = fecha_corte.strftime("%d de %B de %Y")
        else:
            if isinstance(rango, tuple) and len(rango) == 2:
                ini, fin = rango
                mask = (df_corte["Fecha_dt"].dt.date >= ini) & (df_corte["Fecha_dt"].dt.date <= fin)
                df_corte = df_corte[mask]
                titulo_rango = f"{ini.strftime('%d/%m/%Y')} — {fin.strftime('%d/%m/%Y')}"
            else:
                titulo_rango = "rango no válido"

        st.markdown(f"### Corte del {titulo_rango}")

        if df_corte.empty:
            st.warning("No hay ventas registradas en el periodo seleccionado.")
        else:
            total_boletos = len(df_corte)
            total_recaudado = float(df_corte["Costo del boleto"].sum())
            total_efectivo = float(
                df_corte[df_corte["Modo de pago"] == "Efectivo"]["Costo del boleto"].sum()
            )
            total_spei = float(
                df_corte[df_corte["Modo de pago"] == "Transferencia SPEI"]["Costo del boleto"].sum()
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🎟️ Boletos vendidos", total_boletos)
            m2.metric("💵 Efectivo", precio_fmt(total_efectivo))
            m3.metric("🏦 SPEI", precio_fmt(total_spei))
            m4.metric("📊 Total recaudado", precio_fmt(total_recaudado))

            st.divider()

            st.subheader("Resumen por vendedor")
            df_cv = df_corte.copy()
            df_cv["Efectivo"] = df_cv.apply(
                lambda r: r["Costo del boleto"] if r["Modo de pago"] == "Efectivo" else 0.0,
                axis=1,
            )
            df_cv["SPEI"] = df_cv.apply(
                lambda r: r["Costo del boleto"] if r["Modo de pago"] == "Transferencia SPEI" else 0.0,
                axis=1,
            )
            resumen_vend = df_cv.groupby("Vendido por:", as_index=False).agg(
                Boletos=("Folio", "count"),
                Efectivo=("Efectivo", "sum"),
                SPEI=("SPEI", "sum"),
                Total=("Costo del boleto", "sum"),
            )
            resumen_vend = resumen_vend.sort_values("Total", ascending=False).reset_index(drop=True)
            st.dataframe(
                resumen_vend, hide_index=True, use_container_width=True,
                column_config={
                    "Efectivo": st.column_config.NumberColumn("Efectivo", format="$%.2f"),
                    "SPEI": st.column_config.NumberColumn("SPEI", format="$%.2f"),
                    "Total": st.column_config.NumberColumn("Total", format="$%.2f"),
                },
            )

            st.divider()
            st.subheader("Resumen por modalidad")
            resumen_mod = df_cv.groupby("Circunstancia", as_index=False).agg(
                Boletos=("Folio", "count"),
                Total=("Costo del boleto", "sum"),
            ).sort_values("Total", ascending=False).reset_index(drop=True)
            st.dataframe(
                resumen_mod, hide_index=True, use_container_width=True,
                column_config={
                    "Total": st.column_config.NumberColumn("Total", format="$%.2f"),
                },
            )

            st.divider()

            with st.expander("📄 Ver detalle de ventas del periodo"):
                df_detalle = df_corte[[
                    "Folio", "Nombre", "Costo del boleto", "Modo de pago",
                    "Vendido por:", "Fecha", "Boleto digital", "Circunstancia",
                    "Estado",
                ]].rename(columns={"Costo del boleto": "Costo"})
                st.dataframe(
                    df_detalle, hide_index=True, use_container_width=True,
                    column_config={
                        "Folio": st.column_config.NumberColumn("Folio", format="%d"),
                        "Costo": st.column_config.NumberColumn("Costo", format="$%.2f"),
                    },
                )

            st.divider()

            st.subheader("📤 Exportar corte")
            ce1, ce2 = st.columns(2)

            buffer_resumen = io.BytesIO()
            with pd.ExcelWriter(buffer_resumen, engine="openpyxl") as writer:
                resumen_vend.to_excel(writer, index=False, sheet_name="Por vendedor")
                resumen_mod.to_excel(writer, index=False, sheet_name="Por modalidad")
                df_corte[[
                    "Folio", "Nombre", "Costo del boleto", "Modo de pago",
                    "Vendido por:", "Fecha", "Boleto digital", "Circunstancia",
                    "Comentarios", "Estado", "Check-in",
                ]].to_excel(writer, index=False, sheet_name="Detalle")

            nombre_archivo = (
                f"corte_caja_{fecha_corte.strftime('%Y%m%d')}"
                if modo_vista == "Un día"
                else f"corte_caja_{titulo_rango.replace('/', '').replace(' ', '').replace('—', '_')}"
            )

            with ce1:
                st.download_button(
                    "⬇️ Descargar Corte (Excel)",
                    data=buffer_resumen.getvalue(),
                    file_name=f"{nombre_archivo}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            csv_resumen = resumen_vend.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
            with ce2:
                st.download_button(
                    "⬇️ Descargar Resumen (CSV)",
                    data=csv_resumen,
                    file_name=f"{nombre_archivo}_resumen.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ============================================================
# TAB 6: EDITAR / ELIMINAR
# ============================================================
with tab_edit:
    st.subheader("✏️ Editar o eliminar folio")

    if df.empty:
        st.info("No hay ventas registradas.")
    else:
        folio_options = df["Folio"].tolist()
        folio_labels = {
            int(row["Folio"]): f"{int(row['Folio'])} — {row['Nombre']} ({row['Fecha']})"
            for _, row in df.iterrows()
        }

        folio_sel = st.selectbox(
            "Selecciona el folio",
            options=folio_options,
            format_func=lambda f: folio_labels.get(int(f), str(f)),
        )
        idx_sel = df[df["Folio"] == folio_sel].index[0]
        row = df.loc[idx_sel]

        try:
            fecha_default = datetime.strptime(str(row["Fecha"]), "%Y-%m-%d").date()
        except Exception:
            fecha_default = date.today()

        with st.form("form_edit"):
            c1, c2 = st.columns(2)
            with c1:
                nombre_e = st.text_input("Nombre", value=str(row["Nombre"]))
                costo_e = st.number_input(
                    "Costo unitario ($)",
                    min_value=0.0, value=float(row["Costo del boleto"]), step=5.0,
                )
                st.number_input(
                    "Cantidad de boletos",
                    min_value=1, value=1, step=1, disabled=True,
                    help="Cada fila representa 1 boleto.",
                )
                pago_e = st.selectbox(
                    "Modo de pago", MODOS_PAGO,
                    index=MODOS_PAGO.index(row["Modo de pago"]) if row["Modo de pago"] in MODOS_PAGO else 0,
                )
                estado_e = st.selectbox(
                    "Estado del boleto",
                    options=ESTADOS_BOLETO,
                    index=ESTADOS_BOLETO.index(row["Estado"])
                          if row["Estado"] in ESTADOS_BOLETO else 0,
                    help="Válido = aún no entra · Usado = ya entró · Cancelado = anulado",
                )
            with c2:
                vend_e = st.selectbox(
                    "Vendido por:", VENDEDORES,
                    index=VENDEDORES.index(row["Vendido por:"]) if row["Vendido por:"] in VENDEDORES else 0,
                )
                fecha_e = st.date_input("Fecha", value=fecha_default)
                tel_e = st.text_input("Teléfono", value=str(row["Teléfono de contacto"]))
                email_e = st.text_input("e-mail", value=str(row["e-mail"]))
                circ_e = st.selectbox(
                    "Circunstancia", list(MODALIDADES.keys()),
                    index=list(MODALIDADES.keys()).index(row["Circunstancia"])
                    if row["Circunstancia"] in MODALIDADES else 0,
                )

            digital_e = st.checkbox(
                "🎟️ Cliente solicitó boleto digital (PDF)",
                value=(str(row["Boleto digital"]).strip() == "Sí"),
            )

            com_e = st.text_area("Comentarios", value=str(row["Comentarios"]), height=80)

            st.caption(
                f"Folio: **{int(row['Folio'])}**  ·  "
                f"Check-in: **{str(row['Check-in']) or '—'}**"
            )

            ca, cb = st.columns(2)
            guardar = ca.form_submit_button("💾 Guardar cambios", use_container_width=True)
            eliminar = cb.form_submit_button("🗑️ Eliminar folio", use_container_width=True)

        if guardar:
            df.loc[idx_sel, [
                "Nombre", "Costo del boleto", "Modo de pago",
                "Vendido por:", "Fecha", "Teléfono de contacto",
                "e-mail", "Boleto digital", "Circunstancia", "Comentarios",
                "Estado",
            ]] = [
                nombre_e, float(costo_e), pago_e, vend_e,
                fecha_e.strftime("%Y-%m-%d"), tel_e, email_e,
                "Sí" if digital_e else "No", circ_e, com_e,
                estado_e,
            ]
            # Si el nuevo estado no es "Usado", limpiamos el check-in
            if estado_e != "Usado":
                df.loc[idx_sel, "Check-in"] = ""
            guardar_ventas(df)
            st.toast("💾 Folio actualizado.")
            st.rerun()

        if eliminar:
            st.session_state.confirm_delete = folio_sel

        if st.session_state.confirm_delete == folio_sel:
            st.warning(
                f"⚠️ ¿Seguro que deseas eliminar el folio **{int(row['Folio'])}** "
                f"(**{row['Nombre']}**)? El folio quedará libre."
            )
            cc1, cc2 = st.columns(2)
            if cc1.button("Sí, eliminar", type="primary", use_container_width=True):
                df = df.drop(idx_sel).reset_index(drop=True)
                guardar_ventas(df)
                recalcular_next_folio(df, cfg)
                st.session_state.confirm_delete = None
                st.toast(f"🗑️ Folio {int(row['Folio'])} eliminado.")
                st.rerun()
            if cc2.button("Cancelar", use_container_width=True):
                st.session_state.confirm_delete = None
                st.rerun()

        # --- Generar / descargar boleto PDF de cualquier folio ---
        st.divider()
        st.subheader("🎟️ Generar / descargar boleto digital (PDF)")
        st.caption(
            "Útil cuando el cliente pide su boleto digital después de la venta "
            "(aunque no lo haya solicitado al momento de comprar)."
        )
        folio_pdf_sel = st.selectbox(
            "Selecciona el folio para generar el PDF",
            options=df["Folio"].tolist(),
            format_func=lambda f: (
                f"{int(f)} — {df.loc[df['Folio']==f, 'Nombre'].iloc[0]} "
                f"({df.loc[df['Folio']==f, 'Fecha'].iloc[0]})"
            ),
            key="pdf_folio_sel",
        )
        if folio_pdf_sel is not None:
            fila_pdf = df[df["Folio"] == folio_pdf_sel].iloc[0].to_dict()
            try:
                pdf_bytes = generar_boleto_pdf(fila_pdf)
                st.download_button(
                    label=f"📥 Descargar boleto del folio {int(folio_pdf_sel)}",
                    data=pdf_bytes,
                    file_name=(
                        f"boleto_folio_{int(folio_pdf_sel)}_"
                        f"{str(fila_pdf['Nombre']).replace(' ', '_')}.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                    key="pdf_desde_editar",
                )
            except Exception as e:
                st.error(f"Error generando PDF: {e}")


# ============================================================
# TAB 7: EXPORTAR
# ============================================================
with tab_exp:
    st.subheader("📤 Exportar datos")

    if df.empty:
        st.info("No hay datos para exportar.")
    else:
        df_exp = df.copy()
        df_exp["Total de venta"] = df_exp["Costo del boleto"]
        cols = [
            "Folio", "Nombre", "Costo del boleto", "Cantidad de boletos",
            "Total de venta", "Modo de pago", "Vendido por:", "Fecha",
            "Teléfono de contacto", "e-mail", "Boleto digital",
            "Circunstancia", "Comentarios", "Estado", "Check-in",
        ]
        df_exp = df_exp[cols]

        csv_bytes = df_exp.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV",
            data=csv_bytes,
            file_name=f"ventas_boletos_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_exp.to_excel(writer, index=False, sheet_name="Ventas")

            df_res = df.copy()
            df_res["Total"] = df_res["Costo del boleto"]

            df_res.groupby("Vendido por:", as_index=False).agg(
                Boletos=("Folio", "count"),
                Recaudado=("Total", "sum"),
            ).round(2).to_excel(writer, index=False, sheet_name="Resumen por vendedor")

            df_res.groupby("Circunstancia", as_index=False).agg(
                Boletos=("Folio", "count"),
                Recaudado=("Total", "sum"),
            ).round(2).to_excel(writer, index=False, sheet_name="Resumen por modalidad")

        st.download_button(
            "⬇️ Descargar Excel (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"ventas_boletos_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        st.caption("El Excel incluye 3 hojas: Ventas, Resumen por vendedor y Resumen por modalidad.")
