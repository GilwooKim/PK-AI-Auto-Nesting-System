import os
import io
import math
import threading
import time
import xml.etree.ElementTree as ET

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely import affinity
from shapely.geometry import Polygon

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Auto Nesting System",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – dark-mode premium look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* ── Cards / containers ── */
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .label { font-size: 0.78rem; color: #90caf9; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }
.metric-card .value { font-size: 1.9rem; font-weight: 700; color: #ffffff; margin-top: 4px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(102,126,234,.35);
}
.hero h1 { font-size: 2rem; font-weight: 700; color: #fff; margin: 0 0 8px; }
.hero p  { color: rgba(255,255,255,.80); font-size: 1rem; margin: 0; }

/* ── Step badges ── */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff;
    border-radius: 50%;
    width: 28px; height: 28px;
    line-height: 28px;
    text-align: center;
    font-weight: 700;
    margin-right: 8px;
    font-size: .85rem;
}

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg,#667eea,#764ba2) !important; border-radius: 6px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    transition: all .2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102,126,234,.5);
}

/* ── Section headers ── */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #90caf9;
    border-bottom: 1px solid rgba(144,202,249,.25);
    padding-bottom: 6px;
    margin: 18px 0 12px;
    letter-spacing: .04em;
}

/* ── Status chips ── */
.chip {
    display:inline-block; border-radius:20px; padding:3px 12px;
    font-size:.78rem; font-weight:600; margin:2px;
}
.chip-green  { background:rgba(76,175,80,.2);  color:#81c784; border:1px solid #81c784; }
.chip-blue   { background:rgba(33,150,243,.2); color:#64b5f6; border:1px solid #64b5f6; }
.chip-orange { background:rgba(255,152,0,.2);  color:#ffb74d; border:1px solid #ffb74d; }
.chip-red    { background:rgba(244,67,54,.2);  color:#e57373; border:1px solid #e57373; }

/* ── Input labels ── */
label { color: #b0bec5 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        xml_data=[],
        bom_data=pd.DataFrame(),
        merged_data=pd.DataFrame(),
        placed_items=[],
        unplaced_items=[],
        scale_factor=1.0,
        nesting_running=False,
        nesting_done=False,
        cancel_requested=False,
        progress=0.0,
        used_length=0.0,
        efficiency=0.0,
        status_msg="",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Nesting algorithm  (identical logic to the original, runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────
class NestingAlgorithm:
    def __init__(self, patterns, fabric_width=150, max_length=500):
        self.patterns = list(patterns)
        self.fabric_width = fabric_width
        self.max_length = max_length
        self.placed_items = []
        self.unplaced_items = []

    def run(self, progress_cb=None, cancel_flag=None):
        self.patterns.sort(
            key=lambda x: (
                x["poly"].area,
                max(
                    x["poly"].bounds[2] - x["poly"].bounds[0],
                    x["poly"].bounds[3] - x["poly"].bounds[1],
                ),
            ),
            reverse=True,
        )
        placed_polys, placed_bounds = [], []
        step_x = step_y = 3.0
        total = len(self.patterns)

        variants = [(a, m) for a in (0, 90, 180, 270) for m in (1, -1)]

        for idx, item in enumerate(self.patterns):
            if cancel_flag and cancel_flag():
                break

            style = item["style"]
            base_poly = item["poly"]
            best_score = float("inf")
            best_placement = None
            start_y = 0 if style.endswith("_A") or style == "Style_A" else int(self.fabric_width / 2)

            for angle, mirror in variants:
                if cancel_flag and cancel_flag():
                    break

                vpoly = affinity.rotate(base_poly, angle, origin="centroid")
                if mirror == -1:
                    vpoly = affinity.scale(vpoly, xfact=-1.0, origin="centroid")

                mnx, mny, mxx, mxy = vpoly.bounds
                vpoly = affinity.translate(vpoly, xoff=-mnx, yoff=-mny)
                w, h = mxx - mnx, mxy - mny

                if w > self.max_length or h > self.fabric_width:
                    continue

                found_variant = False
                x_range = int(math.ceil((self.max_length - w) / step_x))

                for xi in range(x_range + 1):
                    x = xi * step_x
                    if x > best_score:
                        break

                    if start_y > 0:
                        yc = [yy * step_y for yy in range(int(start_y / step_y), int((self.fabric_width - h) / step_y) + 1)]
                        yc += [yy * step_y for yy in range(0, int(start_y / step_y))]
                    else:
                        yc = [yy * step_y for yy in range(0, int((self.fabric_width - h) / step_y) + 1)]

                    for y in yc:
                        cp = affinity.translate(vpoly, xoff=x, yoff=y)
                        cb = cp.bounds
                        overlap = False
                        for i, pb in enumerate(placed_bounds):
                            if not (cb[2] <= pb[0] or cb[0] >= pb[2] or cb[3] <= pb[1] or cb[1] >= pb[3]):
                                if cp.intersects(placed_polys[i]):
                                    overlap = True
                                    break
                        if not overlap:
                            # gravity compaction
                            moved = True
                            while moved:
                                moved = False
                                t = affinity.translate(cp, xoff=0, yoff=-0.5)
                                tb = t.bounds
                                if tb[1] >= 0:
                                    ov = False
                                    for i, pb in enumerate(placed_bounds):
                                        if not (tb[2] <= pb[0] or tb[0] >= pb[2] or tb[3] <= pb[1] or tb[1] >= pb[3]):
                                            if t.intersects(placed_polys[i]):
                                                ov = True; break
                                    if not ov:
                                        cp = t; moved = True
                                t = affinity.translate(cp, xoff=-0.5, yoff=0)
                                tb = t.bounds
                                if tb[0] >= 0:
                                    ov = False
                                    for i, pb in enumerate(placed_bounds):
                                        if not (tb[2] <= pb[0] or tb[0] >= pb[2] or tb[3] <= pb[1] or tb[1] >= pb[3]):
                                            if t.intersects(placed_polys[i]):
                                                ov = True; break
                                    if not ov:
                                        cp = t; moved = True
                            sc = cp.bounds[2] + cp.bounds[1] * 0.001
                            if sc < best_score:
                                best_score = sc
                                best_placement = {"poly": cp, "bounds": cp.bounds}
                            found_variant = True
                            break
                    if found_variant:
                        break

            if best_placement:
                self.placed_items.append({"style": style, "id": item["id"], "poly": best_placement["poly"]})
                placed_polys.append(best_placement["poly"])
                placed_bounds.append(best_placement["bounds"])
            else:
                self.unplaced_items.append(item)

            if progress_cb:
                progress_cb(list(self.placed_items), idx + 1, total)

        return self.placed_items, self.unplaced_items


# ─────────────────────────────────────────────────────────────────────────────
# Helper: parse XML bytes
# ─────────────────────────────────────────────────────────────────────────────
def parse_xml_bytes(raw_bytes, filename, scale_factor):
    result = []
    file_style_code = filename[0:7]
    try:
        root = ET.fromstring(raw_bytes)
        for piece in root.findall(".//PIECE"):
            unique_elem = piece.find("UNIQUE")
            if unique_elem is None or not unique_elem.text:
                continue
            pattern_id = unique_elem.text.strip()
            best_coords = []

            for polyline in piece.findall(".//polyline"):
                pts_str = polyline.get("points", "")
                pts = []
                for pt in pts_str.strip().split():
                    if "," in pt:
                        try:
                            pts.append((float(pt.split(",")[0]), float(pt.split(",")[1])))
                        except ValueError:
                            pass
                if len(pts) > len(best_coords):
                    best_coords = pts

            for path in piece.findall(".//path"):
                d = path.get("d", "")
                tokens = d.replace("M", " ").replace("L", " ").replace("Z", " ").split()
                pts = []
                for tok in tokens:
                    if "," in tok:
                        try:
                            pts.append((float(tok.split(",")[0]), float(tok.split(",")[1])))
                        except ValueError:
                            pass
                if len(pts) > len(best_coords):
                    best_coords = pts

            if len(best_coords) >= 3:
                poly = Polygon(best_coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda p: p.area)
                poly = affinity.scale(poly, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
                result.append({"FILE_STYLE": file_style_code, "id": pattern_id, "poly": poly})
    except Exception:
        pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw Plotly figure
# ─────────────────────────────────────────────────────────────────────────────
COLORS = [
    "rgba(100,149,237,0.65)", "rgba(80,200,120,0.65)", "rgba(255,127,80,0.65)",
    "rgba(255,215,0,0.65)",   "rgba(218,112,214,0.65)", "rgba(64,224,208,0.65)",
    "rgba(255,182,193,0.65)", "rgba(152,251,152,0.65)",
]


def build_figure(placed_items, fabric_width, min_length, max_length, layers):
    unique_styles = list(dict.fromkeys(i["style"] for i in placed_items))
    style_color = {s: COLORS[i % len(COLORS)] for i, s in enumerate(unique_styles)}
    line_color  = {s: COLORS[i % len(COLORS)].replace("0.65", "1") for i, s in enumerate(unique_styles)}

    fig = go.Figure()

    # fabric boundary
    fig.add_shape(type="rect", x0=0, y0=0, x1=max_length, y1=fabric_width,
                  line=dict(color="rgba(255,255,255,0.5)", width=2),
                  fillcolor="rgba(255,255,255,0.03)")

    # min-length line
    if min_length > 0:
        fig.add_vline(x=min_length, line_dash="dash", line_color="#ff6b6b",
                      annotation_text=f"Min {min_length}cm", annotation_position="top right")

    # pieces
    for item in placed_items:
        poly = item["poly"]
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        xs, ys = poly.exterior.xy
        xs, ys = list(xs), list(ys)
        st_name = item["style"]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            fill="toself",
            fillcolor=style_color.get(st_name, "rgba(160,160,160,0.5)"),
            line=dict(color=line_color.get(st_name, "white"), width=1),
            mode="lines",
            name=st_name,
            showlegend=False,
            hovertemplate=f"<b>ID:</b> {item['id']}<br><b>Style:</b> {st_name}<extra></extra>",
        ))
        cx, cy = poly.centroid.x, poly.centroid.y
        fig.add_annotation(x=cx, y=cy, text=str(item["id"]),
                           showarrow=False, font=dict(size=8, color="white"),
                           bgcolor="rgba(0,0,0,0.45)", borderpad=2)

    # legend manual
    for st_name in unique_styles:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=14, color=style_color[st_name], symbol="square"),
            name=st_name,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,12,41,0.6)",
        font=dict(color="#e0e0e0", family="Inter"),
        xaxis=dict(title="Length (cm)", showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                   zeroline=False, range=[-10, max_length + 20]),
        yaxis=dict(title="Width (cm)", showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                   zeroline=False, range=[-5, fabric_width + 15], scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
        title=dict(
            text=f"Mixed Nesting Marker  ·  Fabric {fabric_width:.1f} cm × Max {max_length:.1f} cm  ·  Layers: {layers}",
            font=dict(size=14, color="#90caf9"),
            x=0.5,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.06)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helper: export XML bytes
# ─────────────────────────────────────────────────────────────────────────────
def build_xml_bytes(placed_items, material, layers, scale_factor):
    root = ET.Element("NestedMarker")
    root.set("material", material)
    root.set("layers", str(layers))
    inv = 1.0 / scale_factor
    for item in placed_items:
        poly = item["poly"]
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        pe = ET.SubElement(root, "PIECE", unique_id=str(item["id"]), style=str(item["style"]))
        for x, y in list(poly.exterior.coords)[:-1]:
            ET.SubElement(pe, "Point", x=str(round(x * inv, 3)), y=str(round(y * inv, 3)))
    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="    ")
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: export PDF bytes (via matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf_bytes(placed_items, fabric_width, min_length, max_length, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(20, 6), facecolor="#0f0c29")
    ax.set_facecolor("#0f0c29")

    unique_styles = list(dict.fromkeys(i["style"] for i in placed_items))
    colors_mpl = ["lightblue","lightgreen","lightsalmon","wheat","plum","cyan","pink","lime"]
    sc = {s: colors_mpl[i % len(colors_mpl)] for i, s in enumerate(unique_styles)}

    for item in placed_items:
        poly = item["poly"]
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.7, fc=sc.get(item["style"], "gray"), ec="white", linewidth=0.5)
        ax.text(poly.centroid.x, poly.centroid.y, str(item["id"]),
                ha="center", va="center", fontsize=5, color="black")

    rect = mpatches.Rectangle((0,0), max_length, fabric_width, linewidth=2, edgecolor="white", facecolor="none")
    ax.add_patch(rect)
    if min_length > 0:
        ax.axvline(x=min_length, color="red", linestyle="--")
    ax.set_xlim(-10, max_length + 20)
    ax.set_ylim(-5, fabric_width + 15)
    ax.set_aspect("equal")
    ax.set_title(f"Nesting Marker | Fabric {fabric_width:.1f}cm × {max_length:.1f}cm | Layers: {layers}",
                 color="white", fontsize=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics calculation
# ─────────────────────────────────────────────────────────────────────────────
def calc_metrics(placed_items, fabric_width):
    if not placed_items:
        return 0.0, 0.0
    used_len = max(i["poly"].bounds[2] for i in placed_items)
    total_area = sum(i["poly"].area for i in placed_items)
    eff = (total_area / (used_len * fabric_width) * 100) if used_len > 0 else 0.0
    return round(used_len, 2), round(eff, 2)


# ─────────────────────────────────────────────────────────────────────────────
# ── UI ────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
  <h1>✂️ AI Auto Nesting System</h1>
  <p>이종 스타일 통합 자동 네스팅 · 고효율 AI 배치 90% 목표 · XML &amp; PDF 출력 지원</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 원단 규격 / 연단 설정")
    st.markdown('<div class="section-header">Fabric Dimensions &amp; Layers</div>', unsafe_allow_html=True)

    xml_unit = st.selectbox("XML 원본 단위", ["mm", "cm", "inch"], index=0, key="xml_unit")
    fabric_width_val = st.number_input("원단 폭 (Width)", min_value=1.0, value=150.0, step=1.0)
    width_unit = st.selectbox("폭 단위", ["cm", "inch"], index=0)
    min_length = st.number_input("최소 길이 (Min Length) cm", min_value=0.0, value=0.0, step=1.0)
    max_length = st.number_input("최대 길이 (Max Length) cm", min_value=1.0, value=500.0, step=10.0)
    layers = st.number_input("원단 겹수 (Layers)", min_value=1, value=1, step=1)

    fabric_width_cm = fabric_width_val * 2.54 if width_unit == "inch" else fabric_width_val

    st.markdown("---")
    st.markdown("### 📋 빠른 도움말")
    with st.expander("사용 방법 보기"):
        st.markdown("""
1. **XML 업로드** – Caligola 패턴 XML 파일들을 업로드합니다.
2. **BOM 업로드** – Excel BOM 파일들을 업로드합니다.
3. **데이터 연동** – 파일명 + UNIQUE ID 기준으로 매핑합니다.
4. **자재 선택** – 네스팅할 자재를 선택하고 스타일별 수량을 입력합니다.
5. **네스팅 실행** – AI가 최적 배치를 계산합니다.
6. **결과 내보내기** – XML 또는 PDF로 저장합니다.
        """)


# ── STEP 1-2-3: File upload & merge ──────────────────────────────────────────
st.markdown('<div class="section-header"><span class="step-badge">1</span> 파일 업로드 &amp; 데이터 연동</div>', unsafe_allow_html=True)

col_xml, col_excel, col_merge = st.columns([2, 2, 1])

with col_xml:
    xml_files = st.file_uploader(
        "Caligola XML 다중 업로드", type=["xml"],
        accept_multiple_files=True, key="xml_uploader",
        help="패턴 조각이 담긴 Caligola XML 파일을 선택하세요."
    )
    if xml_files:
        if st.button("✅ XML 파일 적용", use_container_width=True):
            scale = {"mm": 0.1, "cm": 1.0, "inch": 2.54}[xml_unit]
            st.session_state.scale_factor = scale
            all_data = []
            for f in xml_files:
                all_data += parse_xml_bytes(f.read(), f.name, scale)
            st.session_state.xml_data = all_data
            st.session_state.merged_data = pd.DataFrame()
            st.session_state.placed_items = []
            st.success(f"✔ {len(xml_files)}개 파일 / 총 **{len(all_data)}** 조각 로드 완료")

with col_excel:
    excel_files = st.file_uploader(
        "Excel BOM 다중 업로드", type=["xlsx"],
        accept_multiple_files=True, key="excel_uploader",
        help="BOM 정보가 담긴 Excel 파일을 선택하세요."
    )
    if excel_files:
        if st.button("✅ BOM 파일 적용", use_container_width=True):
            dfs = []
            for f in excel_files:
                try:
                    code = f.name[8:15]
                    df = pd.read_excel(f)
                    df["FILE_STYLE"] = code
                    dfs.append(df)
                except Exception:
                    pass
            if dfs:
                st.session_state.bom_data = pd.concat(dfs, ignore_index=True)
                st.success(f"✔ BOM 로드 완료 ({len(st.session_state.bom_data)} 행)")

with col_merge:
    st.write("")
    st.write("")
    merge_clicked = st.button("🔗 파일명 & UNIQUE 연동", use_container_width=True, type="primary")

if merge_clicked:
    xml_d = st.session_state.xml_data
    bom = st.session_state.bom_data

    if not xml_d or bom.empty:
        st.warning("XML 데이터와 BOM 데이터를 먼저 업로드하세요.")
    else:
        bom = bom.copy()
        bom.columns = bom.columns.str.strip().str.upper()
        bom = bom.loc[:, ~bom.columns.duplicated()]

        rmap = {}
        found_u, found_i, found_n = False, False, False
        for col in bom.columns:
            if not found_u and col in {"PIECE UNIQUE","PIECE_UNIQUE","PIECEUNIQUE","PATTERN ID","PATTERN_ID","UNIQUE ID","UNIQUE_ID"}:
                rmap[col] = "UNIQUE_ID"; found_u = True
            elif not found_i and col in {"ITEM CODE","ITEM_CODE","ITEMCODE"}:
                rmap[col] = "ITEM_CODE"; found_i = True
            elif not found_n and col in {"ITEM NAME","ITEM_NAME","ITEMNAME","MATERIAL NAME","자재명","MATERIAL","ITEM"}:
                rmap[col] = "ITEM_NAME"; found_n = True
        if rmap:
            bom.rename(columns=rmap, inplace=True)

        if "UNIQUE_ID" not in bom.columns or "ITEM_CODE" not in bom.columns:
            st.error("BOM에 UNIQUE_ID 또는 ITEM_CODE 컬럼이 없습니다.")
        else:
            if "ITEM_NAME" not in bom.columns:
                bom["ITEM_NAME"] = ""
            bom["STYLE"] = bom["FILE_STYLE"]
            bom["DISPLAY_MAT"] = bom.apply(
                lambda r: f"{r['ITEM_CODE']} - {r['ITEM_NAME']}"
                if pd.notna(r.get("ITEM_NAME")) and str(r.get("ITEM_NAME","")).strip() != ""
                else str(r["ITEM_CODE"]),
                axis=1,
            )
            df_xml = pd.DataFrame(xml_d).rename(columns={"id": "UNIQUE_ID"})
            merged = pd.merge(bom, df_xml, on=["FILE_STYLE", "UNIQUE_ID"], how="inner")
            st.session_state.merged_data = merged
            st.session_state.bom_data = bom
            st.success(f"✔ 연동 완료! 총 **{len(merged)}** 조각 매핑됨")


# ── STEP 4: Material & quantity ───────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header"><span class="step-badge">2</span> 자재 선택 &amp; 수량 입력</div>', unsafe_allow_html=True)

merged = st.session_state.merged_data
qty_values = {}

if merged.empty:
    st.info("먼저 데이터를 연동해 주세요.")
else:
    materials = merged["DISPLAY_MAT"].dropna().unique().tolist()
    selected_mat = st.selectbox("네스팅할 자재 (코드 - 이름)", materials, key="selected_mat")

    if selected_mat:
        df_mat = merged[merged["DISPLAY_MAT"] == selected_mat]
        style_counts = df_mat.groupby("STYLE").size()

        col_table, col_qty = st.columns([3, 2])

        with col_table:
            st.markdown("**스타일 목록 (포함 조각 수)**")
            style_df = style_counts.reset_index()
            style_df.columns = ["스타일", "조각 수"]
            st.dataframe(style_df, use_container_width=True, hide_index=True)

        with col_qty:
            st.markdown("**스타일별 생산 수량**")
            for style in style_counts.index:
                qty_values[style] = st.number_input(
                    style, min_value=1, value=1, step=1, key=f"qty_{style}"
                )


# ── STEP 5: Run nesting ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header"><span class="step-badge">3</span> 네스팅 실행</div>', unsafe_allow_html=True)

col_run, col_cancel, col_spacer = st.columns([1, 1, 4])

run_clicked    = col_run.button("▶ 네스팅 실행 (Run)", use_container_width=True, type="primary")
cancel_clicked = col_cancel.button("⏹ 작업 취소", use_container_width=True)

if cancel_clicked:
    st.session_state.cancel_requested = True

if run_clicked and not merged.empty and selected_mat:
    patterns_to_nest = []
    df_mat = merged[merged["DISPLAY_MAT"] == selected_mat]
    for _, row in df_mat.iterrows():
        style = row["STYLE"]
        oqty = qty_values.get(style, 1)
        marker_qty = math.ceil(oqty / layers)
        for _ in range(marker_qty):
            patterns_to_nest.append({"style": style, "id": row["UNIQUE_ID"], "poly": row["poly"]})

    if not patterns_to_nest:
        st.warning("배치할 조각이 없습니다.")
    else:
        st.session_state.cancel_requested = False
        st.session_state.nesting_done = False
        st.session_state.placed_items = []
        st.session_state.unplaced_items = []

        # ── Run synchronously with live progress in Streamlit ──────────────
        progress_bar = st.progress(0, text="네스팅 시작...")
        status_text  = st.empty()
        chart_slot   = st.empty()

        algo = NestingAlgorithm(patterns_to_nest, fabric_width=fabric_width_cm, max_length=max_length)

        # We collect results step by step (Streamlit runs synchronously in the main thread)
        placed_polys, placed_bounds = [], []
        total = len(algo.patterns)
        algo.patterns.sort(
            key=lambda x: (
                x["poly"].area,
                max(x["poly"].bounds[2]-x["poly"].bounds[0], x["poly"].bounds[3]-x["poly"].bounds[1]),
            ),
            reverse=True,
        )

        current_placed = []
        current_unplaced = []
        step_x = step_y = 3.0
        variants = [(a, m) for a in (0,90,180,270) for m in (1,-1)]

        for idx, item in enumerate(algo.patterns):
            if st.session_state.cancel_requested:
                break

            style     = item["style"]
            base_poly = item["poly"]
            best_score = float("inf")
            best_place = None
            start_y = 0 if style.endswith("_A") or style == "Style_A" else int(fabric_width_cm / 2)

            for angle, mirror in variants:
                vpoly = affinity.rotate(base_poly, angle, origin="centroid")
                if mirror == -1:
                    vpoly = affinity.scale(vpoly, xfact=-1.0, origin="centroid")
                mnx, mny, mxx, mxy = vpoly.bounds
                vpoly = affinity.translate(vpoly, xoff=-mnx, yoff=-mny)
                w, h = mxx - mnx, mxy - mny
                if w > max_length or h > fabric_width_cm:
                    continue
                found_v = False
                for xi in range(int(math.ceil((max_length - w) / step_x)) + 1):
                    x = xi * step_x
                    if x > best_score:
                        break
                    yc = []
                    if start_y > 0:
                        yc  = [yy*step_y for yy in range(int(start_y/step_y), int((fabric_width_cm-h)/step_y)+1)]
                        yc += [yy*step_y for yy in range(0, int(start_y/step_y))]
                    else:
                        yc  = [yy*step_y for yy in range(0, int((fabric_width_cm-h)/step_y)+1)]
                    for y in yc:
                        cp = affinity.translate(vpoly, xoff=x, yoff=y)
                        cb = cp.bounds
                        overlap = any(
                            not (cb[2]<=pb[0] or cb[0]>=pb[2] or cb[3]<=pb[1] or cb[1]>=pb[3])
                            and cp.intersects(placed_polys[i])
                            for i, pb in enumerate(placed_bounds)
                        )
                        if not overlap:
                            moved = True
                            while moved:
                                moved = False
                                t = affinity.translate(cp, xoff=0, yoff=-0.5)
                                tb = t.bounds
                                if tb[1] >= 0:
                                    ov = any(
                                        not (tb[2]<=pb[0] or tb[0]>=pb[2] or tb[3]<=pb[1] or tb[1]>=pb[3])
                                        and t.intersects(placed_polys[i])
                                        for i, pb in enumerate(placed_bounds)
                                    )
                                    if not ov:
                                        cp = t; moved = True
                                t = affinity.translate(cp, xoff=-0.5, yoff=0)
                                tb = t.bounds
                                if tb[0] >= 0:
                                    ov = any(
                                        not (tb[2]<=pb[0] or tb[0]>=pb[2] or tb[3]<=pb[1] or tb[1]>=pb[3])
                                        and t.intersects(placed_polys[i])
                                        for i, pb in enumerate(placed_bounds)
                                    )
                                    if not ov:
                                        cp = t; moved = True
                            sc = cp.bounds[2] + cp.bounds[1] * 0.001
                            if sc < best_score:
                                best_score = sc
                                best_place = {"poly": cp, "bounds": cp.bounds}
                            found_v = True
                            break
                    if found_v:
                        break

            if best_place:
                current_placed.append({"style": style, "id": item["id"], "poly": best_place["poly"]})
                placed_polys.append(best_place["poly"])
                placed_bounds.append(best_place["bounds"])
            else:
                current_unplaced.append(item)

            pct = (idx + 1) / total
            used_l, eff = calc_metrics(current_placed, fabric_width_cm)
            progress_bar.progress(pct, text=f"배치 중... {idx+1}/{total}  —  사용 길이: {used_l:.1f}cm  효율: {eff:.1f}%")

            # update chart every 5 pieces to avoid slow redraws
            if (idx + 1) % 5 == 0 or idx + 1 == total:
                with chart_slot:
                    st.plotly_chart(
                        build_figure(current_placed, fabric_width_cm, min_length, max_length, layers),
                        use_container_width=True,
                        config={"scrollZoom": True, "displayModeBar": True},
                    )

        # finalise
        st.session_state.placed_items   = current_placed
        st.session_state.unplaced_items = current_unplaced
        st.session_state.nesting_done   = True

        used_l, eff = calc_metrics(current_placed, fabric_width_cm)
        st.session_state.used_length = used_l
        st.session_state.efficiency  = eff

        if st.session_state.cancel_requested:
            progress_bar.progress(pct, text="⛔ 작업이 취소되었습니다.")
        else:
            progress_bar.progress(1.0, text="✅ 네스팅 완료!")

        if current_unplaced:
            st.warning(f"⚠️ {len(current_unplaced)}개의 조각이 캔버스 범위를 초과하여 배치되지 않았습니다.")


# ── STEP 6: Results ───────────────────────────────────────────────────────────
placed = st.session_state.placed_items

if placed:
    used_l, eff = calc_metrics(placed, fabric_width_cm)

    st.markdown("---")
    st.markdown('<div class="section-header"><span class="step-badge">4</span> 네스팅 결과 &amp; 내보내기</div>', unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">배치된 조각</div><div class="value">{len(placed)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">미배치 조각</div><div class="value">{len(st.session_state.unplaced_items)}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="label">사용 길이</div><div class="value">{used_l:.1f} cm</div></div>', unsafe_allow_html=True)
    with c4:
        color = "#4caf50" if eff >= 85 else "#ff9800" if eff >= 70 else "#f44336"
        st.markdown(f'<div class="metric-card"><div class="label">네스팅 효율 (Yield)</div><div class="value" style="color:{color}">{eff:.2f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive chart
    st.plotly_chart(
        build_figure(placed, fabric_width_cm, min_length, max_length, layers),
        use_container_width=True,
        config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False},
        key="final_chart",
    )

    # Download buttons
    st.markdown("#### 📥 결과 내보내기")
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 4])

    xml_bytes = build_xml_bytes(
        placed,
        material=selected_mat if not merged.empty and selected_mat else "",
        layers=layers,
        scale_factor=st.session_state.scale_factor,
    )
    dl_col1.download_button(
        "⬇ XML 내보내기",
        data=xml_bytes,
        file_name="nesting_result.xml",
        mime="application/xml",
        use_container_width=True,
    )

    pdf_bytes = build_pdf_bytes(placed, fabric_width_cm, min_length, max_length, layers)
    dl_col2.download_button(
        "⬇ PDF 내보내기",
        data=pdf_bytes,
        file_name="nesting_result.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    # Placed items table (collapsible)
    with st.expander("📊 배치된 조각 상세 목록"):
        rows = []
        for item in placed:
            b = item["poly"].bounds
            rows.append({
                "Style": item["style"],
                "ID": item["id"],
                "X min (cm)": round(b[0], 2),
                "Y min (cm)": round(b[1], 2),
                "X max (cm)": round(b[2], 2),
                "Y max (cm)": round(b[3], 2),
                "Width (cm)": round(b[2]-b[0], 2),
                "Height (cm)": round(b[3]-b[1], 2),
                "Area (cm²)": round(item["poly"].area, 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
