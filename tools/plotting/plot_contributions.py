"""从 CSV 绘制 Top-5贡献条和全128坐标贡献谱；唯一依赖：matplotlib。

运行：python -m tools.plotting.plot_contributions
也可以：python -m tools.plotting.plot_contributions --csv "path/to/contributions.csv" --output "案例.svg"

字体、加粗、颜色和版式都在下面的“用户设置区”。
Top-5横条从共同起点出发；128行CSV自动按原型编号绘制正负竖条贡献谱，
不显示原型名称，以描边标出已摘取的Top-5。
"""
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator


# ====================== 用户设置区：通常只需改这里 ======================

# ① 数据文件：既支持绝对路径，也支持相对于本脚本的路径。
# 例如：CSV_PATH = r"../../outputs/my_case/contributions.csv"
CSV_PATH = r"../../outputs/synthetic_demo/contributions.csv"

# 输出始终在本脚本所在文件夹。只填文件名，不用填文件夹路径。
OUTPUT_NAME = "contribution_audit.svg"

# ② 字体：改 FONT_SIZE 可整体放大/缩小文字；改 FONT_WEIGHT 可加粗/取消。
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 18                        # 总字号，单位 pt，例如 18、22、26
FONT_WEIGHT = "bold"                 # "bold" = 加粗；"normal" = 不加粗
FONT_STYLE = "normal"                # 保持正体，名称和 P 编号不使用斜体

# 如需单独调节某类文字，把右侧改成数字即可，例如 VALUE_FONT_SIZE = 22。
LABEL_FONT_SIZE = FONT_SIZE           # 左侧原型名称及编号
VALUE_FONT_SIZE = FONT_SIZE           # 右侧带正负号的数值
TICK_FONT_SIZE = FONT_SIZE - 3        # 横轴刻度
AXIS_FONT_SIZE = FONT_SIZE - 2        # 横轴标题
LEGEND_FONT_SIZE = FONT_SIZE - 2      # Positive / Negative 图例

# ③ 显示内容。
TOP_N = 5                            # 5 = Top-5；3 = Top-3；None = 全部行
SORT_BY_ABS = True                   # True = 按贡献绝对值降序；False = 保持 CSV 行序
COLUMN_COUNT = 1                    # 仅用于横条模式；128项自动使用贡献谱
HIGHLIGHT_TOP_N = 0                  # 仅用于横条模式；贡献谱自动描边Top-5
SHOW_LABELS = True                  # 是否显示左侧名称
SHOW_PROTOTYPE_IDS = True           # 是否保留/补充 (P编号)
SHOW_VALUES = True                  # 是否显示右侧数值
SHOW_LEGEND = True                  # 是否显示正、负贡献图例
SHOW_AXIS = True                    # 是否显示横轴及刻度
X_AXIS_LABEL = "Contribution magnitude"
DECIMAL_PLACES = 3                  # 贡献数值保留几位小数
SMALL_VALUES_SCIENTIFIC = True      # 很小的非零项用科学计数法，避免被显示成0.000

# ⑥ 自动识别到128行时，使用论文图2右上角的紧凑正负贡献谱布局。
PROFILE_SIZE_INCHES = (6.4, 2.30)   # 完整128项图幅（英寸）
PROFILE_FONT_SIZE = 9               # 坐标轴和图例字号；不显示原型名称/逐项数值
PROFILE_BAR_WIDTH = 0.82            # 每个原型的横向宽度，原型间保留小间距
PROFILE_REST_ALPHA = 0.38           # 非Top-5条的透明度
PROFILE_TOP5_EDGE_WIDTH = 0.72      # Top-5描边粗细
PROFILE_SHOW_AXIS_LABELS = True    # False时成为与图2一致的无文字迷你谱图
PROFILE_SHOW_LEGEND = True          # 图例说明正负方向及Top-5描边

# 通常不用指定列名，脚本会识别现有 CSV；自定义表头时可手动填写。
# 例如 LABEL_COLUMN = "英文显示名"；VALUE_COLUMN = "贡献q"。
LABEL_COLUMN = None
VALUE_COLUMN = None
ID_COLUMN = None

# ④ 配色：延续论文已有风格。颜色只由“带正负号的贡献值”决定。
POSITIVE_COLOR = "#267D70"           # 正贡献，深绿色
NEGATIVE_COLOR = "#B96050"           # 负贡献，砖红色
ZERO_COLOR = "#83909A"               # 零贡献，灰色
TEXT_COLOR = "#263541"
AXIS_COLOR = "#BAC6CD"

# ⑤ 图幅：单位英寸。较长名称或更大字号会自动扩展画布，避免文字重叠。
FIGURE_WIDTH = 12.0
ROW_HEIGHT = 0.60
BAR_THICKNESS = 0.30
LABEL_BAR_GAP_INCHES = 0.38         # 名称右端到条形零线的最小留白（英寸）
VALUE_BAR_GAP_INCHES = 0.30         # 条形绘图区到右侧数值的留白（英寸）
TRANSPARENT_BACKGROUND = True       # 透明背景，方便放入 PPT/整幅图

# True：SVG 保留可编辑文本；False：文字转为路径，外观更便于跨电脑保持。
# 默认False，避免WPS/PPT导入SVG时替换字体、重排文字而覆盖条形。
# 文字转路径仍是矢量图；要改字号/加粗，在本脚本修改后重新生成。
SVG_EDITABLE_TEXT = False

# =======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
VALUE_COLUMNS = ("贡献q", "signed_margin_contribution", "signed_contribution",
                 "contribution", "q", "贡献值", "贡献", "value")
LABEL_COLUMNS = ("英文显示名", "display_name_en", "label", "name_en", "英文名称",
                 "name", "feature", "形状名称", "名称", "中文显示名")
ID_COLUMNS = ("prototype_id", "原型", "prototype", "原型编号", "id")
TOP5_FLAG_COLUMNS = ("is_top5", "selected_top5", "top5", "top_5")
TOP5_RANK_COLUMNS = ("top5_rank", "top_5_rank", "Top5排名", "Top-5排名")


def resolve_csv_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def choose_column(headers, explicit, candidates, required=False):
    if explicit is not None:
        if explicit not in headers:
            raise ValueError(f"找不到列 {explicit!r}；现有列：{', '.join(headers)}")
        return explicit
    lookup = {h.casefold(): h for h in headers}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    if required:
        raise ValueError("未找到带正负号的贡献列。请在 VALUE_COLUMN 中填写列名。"
                         f"\n现有列：{', '.join(headers)}")
    return None


def prototype_name(value):
    value = str(value).strip()
    match = re.fullmatch(r"[Pp]?\s*(\d+)(?:\.0+)?", value)
    return f"P{int(match.group(1))}" if match else value


def read_contributions(path, profile=False):
    if not path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在：{path}\n请修改开头的 CSV_PATH。")
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV 编码无法识别，请另存为 UTF-8 CSV。")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if reader.fieldnames is None:
        raise ValueError("CSV 没有表头。最简单的表头是：label,contribution")
    reader.fieldnames = [name.strip() for name in reader.fieldnames]
    value_col = choose_column(reader.fieldnames, VALUE_COLUMN, VALUE_COLUMNS, required=True)
    label_col = choose_column(reader.fieldnames, LABEL_COLUMN, LABEL_COLUMNS)
    id_col = choose_column(reader.fieldnames, ID_COLUMN, ID_COLUMNS)
    flag_col = choose_column(reader.fieldnames, None, TOP5_FLAG_COLUMNS)
    top5_rank_col = choose_column(reader.fieldnames, None, TOP5_RANK_COLUMNS)
    if label_col is None and id_col is None:
        raise ValueError("未找到名称或原型编号列，请设置 LABEL_COLUMN 或 ID_COLUMN。")
    if profile and id_col is None:
        raise ValueError("完整128项贡献谱需要 prototype_id 列，以按原型坐标排序。")
    rows = []
    for line, raw_row in enumerate(reader, start=2):
        if not any(str(v or "").strip() for v in raw_row.values()):
            continue
        cell = str(raw_row.get(value_col) or "").strip().replace("\u2212", "-")
        try:
            value = float(cell)
        except ValueError as exc:
            raise ValueError(f"CSV 第 {line} 行的贡献值不是数字：{cell!r}。"
                             "请填带正负号的原始贡献数值，而非绝对值或 mean±std。") from exc
        if not math.isfinite(value):
            raise ValueError(f"CSV 第 {line} 行包含非有限贡献值：{cell!r}")
        label = str(raw_row.get(label_col) or "").strip() if label_col else ""
        pid = prototype_name(raw_row.get(id_col) or "") if id_col else ""
        if profile:
            match = re.fullmatch(r"P(\d+)", pid)
            if match is None:
                raise ValueError(f"CSV 第 {line} 行 prototype_id 无效：{pid!r}")
            prototype_id = int(match.group(1))
        else:
            prototype_id = None
        if not label:
            label = "" if profile else pid
        if not label:
            if not profile:
                raise ValueError(f"CSV 第 {line} 行的名称和编号均为空。")
        if not SHOW_PROTOTYPE_IDS:
            label = re.sub(r"\s*[（(]\s*[Pp]\d+\s*[）)]", "", label).strip() or pid
        elif pid and label != pid and not re.search(rf"(?<![A-Za-z0-9]){re.escape(pid)}(?!\d)", label):
            label = f"{label} ({pid})"
        flag = str(raw_row.get(flag_col) or "").strip().casefold() in {"true", "1", "yes", "y"} if flag_col else False
        rank_cell = str(raw_row.get(top5_rank_col) or "").strip() if top5_rank_col else ""
        rank_marked = rank_cell in {"1", "2", "3", "4", "5"}
        rows.append({"label": label, "value": value, "csv_line": line,
                     "prototype_id": prototype_id, "is_top5": flag or rank_marked})
    if not rows:
        raise ValueError("CSV 没有可绘制的数据行。")
    if profile:
        ids = [row["prototype_id"] for row in rows]
        if sorted(ids) != list(range(128)):
            raise ValueError("128项贡献谱要求prototype_id恰好覆盖0..127，且无重复。")
        rows.sort(key=lambda row: row["prototype_id"])
        if sum(row["is_top5"] for row in rows) != 5:
            selected = {id(row) for row in sorted(rows, key=lambda row: -abs(row["value"]))[:5]}
            for row in rows:
                row["is_top5"] = id(row) in selected
    elif SORT_BY_ABS:
        rows.sort(key=lambda row: -abs(row["value"]))
    if not profile and TOP_N is not None:
        if not isinstance(TOP_N, int) or TOP_N < 1:
            raise ValueError("TOP_N 应为正整数或 None。")
        rows = rows[:TOP_N]
    return rows


def count_data_rows(path):
    """Count nonempty CSV records so a 128-coordinate file selects profile mode."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV编码无法识别，请另存为UTF-8 CSV。")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return sum(1 for row in csv.DictReader(io.StringIO(text), dialect=dialect)
               if any(str(value or "").strip() for value in row.values()))


def contribution_color(value):
    if value > 0:
        return POSITIVE_COLOR
    if value < 0:
        return NEGATIVE_COLOR
    return ZERO_COLOR


def draw_chart(rows):
    # 所有字体均通过这里读取上方设置；bold 会选择真实粗体字体。
    findfont(FontProperties(family=FONT_FAMILY, weight=FONT_WEIGHT), fallback_to_default=False)
    plt.rcParams.update({"font.family": [FONT_FAMILY, "Microsoft YaHei", "DejaVu Sans"],
                         "font.weight": FONT_WEIGHT, "font.style": FONT_STYLE,
                         "svg.fonttype": "none" if SVG_EDITABLE_TEXT else "path",
                         "axes.unicode_minus": False, "text.usetex": False})
    def font(size):
        return FontProperties(family=[FONT_FAMILY, "Microsoft YaHei", "DejaVu Sans"],
                              size=size, weight=FONT_WEIGHT, style=FONT_STYLE)
    fonts = {"label": font(LABEL_FONT_SIZE), "value": font(VALUE_FONT_SIZE),
             "tick": font(TICK_FONT_SIZE), "axis": font(AXIS_FONT_SIZE),
             "legend": font(LEGEND_FONT_SIZE)}
    fig = plt.figure(figsize=(FIGURE_WIDTH, 4), facecolor="white")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    def dimensions(text, fp):
        w, h, descent = renderer.get_text_width_height_descent(text, fp, ismath=False)
        return w/fig.dpi, (h+descent)/fig.dpi
    values = [f'{row["value"]:+.2e}'
              if SMALL_VALUES_SCIENTIFIC and 0 < abs(row["value"]) < .5*10**(-DECIMAL_PLACES)
              else f'{row["value"]:+.{DECIMAL_PLACES}f}' for row in rows]
    label_width = max(dimensions(row["label"], fonts["label"])[0] for row in rows) if SHOW_LABELS else 0
    value_width = max(dimensions(value, fonts["value"])[0] for value in values) if SHOW_VALUES else 0
    max_text_height = max(dimensions(row["label"], fonts["label"])[1] for row in rows) if SHOW_LABELS else 0
    if SHOW_VALUES:
        max_text_height = max(max_text_height, max(dimensions(v, fonts["value"])[1] for v in values))
    row_height = max(ROW_HEIGHT, max_text_height+.10)
    bar_height = min(BAR_THICKNESS, row_height*.70)
    if not isinstance(COLUMN_COUNT, int) or COLUMN_COUNT < 1:
        raise ValueError("COLUMN_COUNT 应为正整数。")
    columns = min(COLUMN_COUNT, len(rows))
    rows_per_column = math.ceil(len(rows)/columns)
    gutter = .35 if columns > 1 else 0
    if LABEL_BAR_GAP_INCHES < 0 or VALUE_BAR_GAP_INCHES < 0:
        raise ValueError("名称/数值与条形之间的留白不能为负。")
    local_left = label_width+.10+LABEL_BAR_GAP_INCHES if SHOW_LABELS else .18
    right_margin = value_width+.10+VALUE_BAR_GAP_INCHES if SHOW_VALUES else .18
    legend = [(POSITIVE_COLOR, "Positive"), (NEGATIVE_COLOR, "Negative")]
    if any(row["value"] == 0 for row in rows):
        legend.append((ZERO_COLOR, "Zero"))
    if HIGHLIGHT_TOP_N:
        legend.append(("#E8EDF2", f"Selected Top-{HIGHLIGHT_TOP_N}"))
    legend_width = sum(.42+dimensions(label, fonts["legend"])[0] for _, label in legend)
    min_bar_width = 1.10 if columns > 1 else 3.4
    min_column_width = local_left+right_margin+min_bar_width
    width = max(FIGURE_WIDTH, columns*min_column_width+(columns-1)*gutter,
                local_left+legend_width+.18 if SHOW_LEGEND else 0)
    column_width = (width-(columns-1)*gutter)/columns
    legend_height = dimensions("Positive", fonts["legend"])[1]+.20 if SHOW_LEGEND else 0
    if SHOW_AXIS:
        bottom = dimensions("0.000", fonts["tick"])[1]+dimensions(X_AXIS_LABEL, fonts["axis"])[1]+.39
    else:
        bottom = .12
    height = bottom+rows_per_column*row_height+legend_height+.12
    fig.set_size_inches(width, height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    maximum = max(abs(row["value"]) for row in rows)
    upper = maximum*1.06 if maximum else 1.0
    text_artists = []
    row_layout_checks = []
    for column in range(columns):
        offset = column*(column_width+gutter)
        left = offset+local_left
        right = offset+column_width-right_margin
        scale = (right-left)/upper
        first = column*rows_per_column
        last = min(len(rows), first+rows_per_column)
        # 每列使用同样的数值尺度；列内所有条形的左端共用零点。
        ax.plot([left, left], [bottom, bottom+rows_per_column*row_height], color=AXIS_COLOR, lw=.8)
        selected_in_column = max(0, min(last, HIGHLIGHT_TOP_N)-first)
        if selected_in_column:
            box = Rectangle((offset+.025, bottom+(rows_per_column-selected_in_column)*row_height),
                            column_width-.05, selected_in_column*row_height,
                            facecolor="#E8EDF2", edgecolor="#83909A", lw=.65, zorder=0)
            box.set_gid("selected-top5-frame")
            ax.add_patch(box)
        for index in range(first, last):
            row, value_label = rows[index], values[index]
            y = bottom+(rows_per_column-(index-first)-.5)*row_height
            color = contribution_color(row["value"])
            bar = Rectangle((left, y-bar_height/2), abs(row["value"])*scale,
                            bar_height, facecolor=color, edgecolor="none", zorder=2)
            bar.set_gid(f'audit-bar-{index+1}')
            ax.add_patch(bar)
            if SHOW_LABELS:
                label_artist = ax.text(offset+.10, y, row["label"], fontproperties=fonts["label"],
                                       ha="left", va="center", color=TEXT_COLOR)
                label_artist.set_gid(f'audit-label-{index+1}')
                text_artists.append(label_artist)
                row_layout_checks.append((label_artist, bar, "left"))
            if SHOW_VALUES:
                value_artist = ax.text(offset+column_width-.10, y, value_label, fontproperties=fonts["value"],
                                       ha="right", va="center", color=color)
                value_artist.set_gid(f'audit-value-{index+1}')
                text_artists.append(value_artist)
                row_layout_checks.append((value_artist, bar, "right"))
        if SHOW_AXIS:
            y = bottom-.09
            ax.plot([left, right], [y, y], color=AXIS_COLOR, lw=.7)
            ticks = [float(t) for t in MaxNLocator(nbins=3 if columns > 1 else 4, steps=[1, 2, 2.5, 5, 10]).tick_values(0, upper) if 0 <= t <= upper]
            step = ticks[1]-ticks[0] if len(ticks) > 1 else upper
            digits = next((n for n in range(7) if math.isclose(round(step, n), step, rel_tol=1e-8)), 6)
            for tick in ticks:
                x = left+tick*scale
                ax.plot([x, x], [y, y-.04], color=AXIS_COLOR, lw=.7)
                text_artists.append(ax.text(x, y-.09, f"{tick:.{digits}f}", fontproperties=fonts["tick"],
                                            ha="center", va="top", color=TEXT_COLOR))
    if SHOW_LEGEND:
        x = local_left
        y = bottom+rows_per_column*row_height+legend_height/2
        for color, label in legend:
            ax.add_patch(Rectangle((x, y-.045), .17, .09, facecolor=color, edgecolor="none"))
            text_artists.append(ax.text(x+.24, y, label, fontproperties=fonts["legend"],
                                        ha="left", va="center", color=TEXT_COLOR))
            x += .42+dimensions(label, fonts["legend"])[0]
    if SHOW_AXIS:
        text_artists.append(ax.text(width/2, .07, X_AXIS_LABEL, fontproperties=fonts["axis"],
                                    ha="center", va="bottom", color=TEXT_COLOR))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in text_artists:
        box = artist.get_window_extent(renderer)
        if not (fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)):
            plt.close(fig)
            raise ValueError(f"文字超出画布：{artist.get_text()}；请增大 FIGURE_WIDTH 或 ROW_HEIGHT。")
    for artist, bar, side in row_layout_checks:
        text_box = artist.get_window_extent(renderer)
        bar_box = bar.get_window_extent(renderer)
        if ((side == "left" and text_box.x1 > bar_box.x0-.02*fig.dpi)
                or (side == "right" and text_box.x0 < bar_box.x1+.02*fig.dpi)):
            plt.close(fig)
            raise ValueError(f"文字与条形间距不足：{artist.get_text()}；请增大相应的GAP_INCHES。")
    return fig


def draw_profile(rows):
    """Draw a name-free, 128-coordinate signed profile like Fig. 2's readout."""
    if len(rows) != 128:
        raise ValueError(f"Shape contribution profile requires exactly 128 rows, got {len(rows)}.")
    rows = sorted(rows, key=lambda row: row["prototype_id"])
    if [row["prototype_id"] for row in rows] != list(range(128)):
        raise ValueError("Profile coordinates must be prototype_id 0..127 with no duplicates.")
    values = [float(row["value"]) for row in rows]
    selected = {row["prototype_id"] for row in rows if row["is_top5"]}
    if len(selected) != 5:
        raise ValueError(f"Profile must highlight exactly five Top-5 rows; got {len(selected)}.")
    findfont(FontProperties(family=FONT_FAMILY, weight=FONT_WEIGHT), fallback_to_default=False)
    plt.rcParams.update({"font.family": [FONT_FAMILY, "Microsoft YaHei", "DejaVu Sans"],
                         "font.weight": FONT_WEIGHT, "font.style": FONT_STYLE,
                         "svg.fonttype": "none" if SVG_EDITABLE_TEXT else "path",
                         "pdf.fonttype": 42, "axes.unicode_minus": False, "text.usetex": False,
                         "path.simplify": False})
    fp = FontProperties(family=[FONT_FAMILY, "Microsoft YaHei", "DejaVu Sans"],
                        size=PROFILE_FONT_SIZE, weight=FONT_WEIGHT, style=FONT_STYLE)
    fig = plt.figure(figsize=PROFILE_SIZE_INCHES, facecolor="white")
    ax = fig.add_axes([.105, .245, .88, .57] if PROFILE_SHOW_AXIS_LABELS
                      else [.025, .06, .95, .88])
    limit = max(abs(value) for value in values)*1.12
    if limit == 0:
        limit = 1.0
    ax.set_xlim(-.7, 127.7)
    ax.set_ylim(-limit, limit)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4EAEE", linewidth=.42, zorder=0)
    ax.axhline(0, color=AXIS_COLOR, linewidth=.72, zorder=1)
    bars = ax.bar(range(128), values, width=PROFILE_BAR_WIDTH,
                  color=[POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR if v < 0 else ZERO_COLOR for v in values],
                  edgecolor=[TEXT_COLOR if pid in selected else "none" for pid in range(128)],
                  linewidth=[PROFILE_TOP5_EDGE_WIDTH if pid in selected else 0 for pid in range(128)],
                  zorder=2, antialiased=False)
    for pid, bar in enumerate(bars):
        bar.set_alpha(1.0 if pid in selected else PROFILE_REST_ALPHA)
        bar.set_gid(f"audit-profile-P{pid:03d}" + ("-top5" if pid in selected else ""))
    if PROFILE_SHOW_AXIS_LABELS:
        ax.set_xticks([0, 32, 64, 96, 127])
        ax.set_xticklabels(["0", "32", "64", "96", "127"], fontproperties=fp)
        ax.set_xlabel("Prototype index", fontproperties=fp, labelpad=2)
        ax.set_ylabel("Signed contribution", fontproperties=fp, labelpad=3)
        ax.tick_params(axis="both", labelsize=PROFILE_FONT_SIZE, length=2.4, width=.55, colors=TEXT_COLOR, pad=2)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(fp)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(AXIS_COLOR)
        ax.spines["bottom"].set_color(AXIS_COLOR)
        ax.spines["left"].set_linewidth(.55)
        ax.spines["bottom"].set_linewidth(.55)
    else:
        ax.set_axis_off()
    if PROFILE_SHOW_LEGEND:
        handles = [Patch(facecolor=POSITIVE_COLOR, alpha=PROFILE_REST_ALPHA, label="Positive"),
                   Patch(facecolor=NEGATIVE_COLOR, alpha=PROFILE_REST_ALPHA, label="Negative"),
                   Patch(facecolor="white", edgecolor=TEXT_COLOR, linewidth=PROFILE_TOP5_EDGE_WIDTH,
                         label="Selected Top-5")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.35),
                  ncol=3, frameon=False, prop=fp, handlelength=1.0,
                  handleheight=.65, columnspacing=1.2, handletextpad=.4, borderaxespad=0)
    fig.canvas.draw()
    if PROFILE_SHOW_AXIS_LABELS:
        renderer = fig.canvas.get_renderer()
        for artist in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label]:
            box = artist.get_window_extent(renderer)
            if not (fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)):
                plt.close(fig)
                raise ValueError(f"Profile label exceeds canvas: {artist.get_text()}")
    return fig


def main():
    global TOP_N, COLUMN_COUNT, HIGHLIGHT_TOP_N
    global PROFILE_SHOW_AXIS_LABELS, PROFILE_SHOW_LEGEND
    global SVG_EDITABLE_TEXT
    parser = argparse.ArgumentParser(description="CSV → Top-5贡献条或128坐标贡献谱（SVG）")
    parser.add_argument("--csv", default=CSV_PATH, help="输入 CSV 路径；也可直接改脚本开头 CSV_PATH")
    parser.add_argument("--output", default=OUTPUT_NAME, help="输出文件名，SVG 始终保存在本脚本所在文件夹")
    parser.add_argument("--all", action="store_true", help="横条模式绘制全部CSV行；128项贡献谱始终完整绘制")
    parser.add_argument("--columns", type=int, help="仅横条模式：分列数量，所有列共用同一数值尺度")
    parser.add_argument("--highlight-top5", action="store_true", help="仅横条模式：用浅灰框标出Top-5；贡献谱自动描边Top-5")
    parser.add_argument("--mode", choices=["auto", "profile", "bars"], default="auto",
                        help="auto会将恰好128行识别为贡献谱；其他CSV绘制贡献条")
    parser.add_argument("--no-text", action="store_true",
                        help="贡献谱隐藏坐标文字和图例，适合像图2一样嵌入框架图")
    parser.add_argument("--editable-text", action="store_true",
                        help="保留SVG文字节点；默认转为矢量轮廓以固定排版，避免导入软件替换字体")
    args = parser.parse_args()
    if args.editable_text:
        SVG_EDITABLE_TEXT = True
    if args.all:
        TOP_N = None
    if args.columns is not None:
        COLUMN_COUNT = args.columns
    if args.highlight_top5:
        HIGHLIGHT_TOP_N = 5
    output = Path(args.output)
    if output.name != args.output or output.suffix.lower() != ".svg":
        parser.error("--output 只填 .svg 文件名，例如 案例.svg；不要填文件夹路径。")
    path = resolve_csv_path(args.csv)
    row_count = count_data_rows(path)
    mode = "profile" if args.mode == "auto" and row_count == 128 else args.mode
    if mode == "profile":
        if row_count != 128:
            parser.error(f"profile模式需要128行CSV，当前是{row_count}行。")
        if args.no_text:
            PROFILE_SHOW_AXIS_LABELS = False
            PROFILE_SHOW_LEGEND = False
        rows = read_contributions(path, profile=True)
        fig = draw_profile(rows)
    else:
        rows = read_contributions(path)
        fig = draw_chart(rows)
    target = SCRIPT_DIR/output.name
    try:
        fig.savefig(target, format="svg", transparent=TRANSPARENT_BACKGROUND,
                    metadata={"Creator": "Matplotlib: signed prototype contribution audit"})
    finally:
        plt.close(fig)
    positive = sum(row["value"] > 0 for row in rows)
    negative = sum(row["value"] < 0 for row in rows)
    print(f"读取：{path}\n绘制：{len(rows)} 项，正贡献 {positive} 项，负贡献 {negative} 项。")
    print(f"绘图模式：{'128项形状贡献谱（按原型编号排序，无名称）' if mode == 'profile' else '独立贡献条'}。")
    if mode == "profile":
        print(f"Top-5描边原型：{', '.join('P'+str(r['prototype_id']) for r in rows if r['is_top5'])}")
        print(f"字体：{FONT_FAMILY}，谱图字号 {PROFILE_FONT_SIZE} pt，字重 {FONT_WEIGHT}。")
    else:
        print(f"字体：{FONT_FAMILY}，总字号 {FONT_SIZE} pt，字重 {FONT_WEIGHT}。")
    print(f"已保存：{target}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
