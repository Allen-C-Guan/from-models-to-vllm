"""Book7 架构样稿制图：中文软件架构图，输出可编辑 SVG 与 PNG。

运行：在工作区根 source ./env.sh 后执行本文件。仅使用 CPU 和现有 matplotlib。
图的机制依据与简化边界见 research/Book7-架构重写/2026-11-02/。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "drafts/Book7-vLLM架构剖析/figures"
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
INK, MUTED = "#1E293B", "#526174"
BLUE, ORANGE, GREEN, PURPLE = "#24659B", "#B4611B", "#20775F", "#7055A1"
FILL = {BLUE: "#ECF4FC", ORANGE: "#FFF4E7", GREEN: "#EAF6F0", PURPLE: "#F3EFFA", MUTED: "#F4F6F9"}
matplotlib.rcParams.update({"svg.fonttype": "none", "axes.unicode_minus": False})


def text(ax, x, y, content, size=17, color=INK, ha="center", weight="normal", bg=False):
    return ax.text(x, y, content, fontproperties=FONT, fontsize=size, color=color,
                   ha=ha, va="center", weight=weight, linespacing=1.55,
                   bbox=dict(facecolor="white", edgecolor="none", pad=3) if bg else None)


def canvas(title, subtitle, height=760, width=1200):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    text(ax, 40, 36, title, 23, ha="left", weight="bold")
    text(ax, 40, 76, subtitle, 14, MUTED, ha="left")
    return fig, ax


def box(ax, x, y, w, h, title, body="", color=BLUE, fill=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                               linewidth=1.6, edgecolor=color, facecolor=fill or FILL[color], zorder=2))
    if body:
        text(ax, x + w / 2, y + h * .30, title, 18, color, weight="bold")
        text(ax, x + w / 2, y + h * .69, body, 14)
    else:
        text(ax, x + w / 2, y + h / 2, title, 18, color, weight="bold")


def group(ax, x, y, w, h, title, color=MUTED, fill="#FAFBFD", dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12",
                               linewidth=1.3, linestyle="--" if dashed else "-",
                               edgecolor=color, facecolor=fill, zorder=-1))
    text(ax, x + 18, y + 25, title, 16, color, ha="left", weight="bold")


def arrow(ax, points, label=None, label_at=None, color=MUTED, dashed=False, size=14):
    # 分段正交路径让拐点和数据方向可检查，不依赖自动布局的隐式布线。
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.8,
                linestyle="--" if dashed else "-", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=14, linewidth=1.8, color=color,
                                linestyle="--" if dashed else "-", zorder=3,
                                shrinkA=1, shrinkB=3))
    if label:
        if label_at is None:
            a, b = points[-2:]
            label_at = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 17)
        text(ax, *label_at, label, size, color, bg=True)


def note(ax, y, content, x=40):
    text(ax, x, y, content, 14, MUTED, ha="left")


def save(fig, slug):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{slug}.svg", metadata={"Date": None})
    fig.savefig(OUT / f"{slug}.png", dpi=180)
    plt.close(fig)
    print(slug)


def boundary():
    fig, ax = canvas("图 1.1  模型在推理系统中的位置", "先看系统边界：生成需求 → 请求组织 → 设备计算 → 用户结果", 760)
    box(ax, 400, 125, 400, 80, "用户的 Python 程序", "提交文本与生成要求", MUTED)
    group(ax, 160, 285, 880, 180, "")
    text(ax, 500, 310, "vLLM 的主机侧程序（运行于 CPU）", 16, MUTED, ha="left", weight="bold")
    box(ax, 195, 350, 235, 85, "输入与输出", "文本适配 · 结果交付", BLUE)
    box(ax, 485, 350, 225, 85, "核心协调", "调度 · KV 管理", ORANGE)
    box(ax, 765, 350, 235, 85, "执行组织", "准备输入 · 提交计算", PURPLE)
    arrow(ax, [(480, 205), (480, 240), (310, 240), (310, 350)], "生成请求", (360, 240), BLUE)
    arrow(ax, [(210, 350), (130, 350), (130, 170), (400, 170)], "最终文本", (240, 170), BLUE)
    arrow(ax, [(430, 375), (485, 375)], color=ORANGE)
    arrow(ax, [(710, 375), (765, 375)], color=PURPLE)
    group(ax, 325, 565, 715, 135, "加速设备（本书以 NPU 为例）", PURPLE, "#FCFAFF")
    box(ax, 350, 620, 180, 55, "模型权重", color=PURPLE)
    box(ax, 570, 620, 210, 55, "模型计算与采样", color=PURPLE)
    box(ax, 820, 620, 190, 55, "KV 张量", color=GREEN)
    arrow(ax, [(915, 435), (915, 565)], "输入／执行指令", (990, 510), PURPLE)
    arrow(ax, [(660, 565), (660, 510), (800, 510), (800, 435)], "计算结果", (720, 510), PURPLE)
    arrow(ax, [(765, 415), (710, 415)], color=PURPLE)
    arrow(ax, [(485, 415), (430, 415)], color=BLUE)
    note(ax, 735, "模型执行只是生成系统的一部分；本图没有规定各组件的进程归属。")
    save(fig, "fig-1-1-boundary")


def components():
    fig, ax = canvas("图 1.2  vLLM 核心职责与协作", "职责图：用数据与控制的交接理解分工，不把方框当成独立进程", 860)
    box(ax, 30, 175, 235, 100, "输入适配", "文本／参数\n→ 计算请求", BLUE)
    group(ax, 325, 125, 540, 445, "EngineCore：组织核心计算循环", ORANGE, "#FFFCF8")
    box(ax, 355, 180, 480, 90, "组织调度、执行与结果更新", "持有 Scheduler 与 Executor · 串起交接", ORANGE)
    box(ax, 355, 390, 205, 120, "调度器", "Scheduler：维护进度\n决定谁算、算多少", ORANGE)
    box(ax, 620, 390, 215, 120, "KV 管理器", "KVCacheManager\n管理块归属与分配", GREEN)
    box(ax, 925, 180, 245, 115, "执行层", "Executor · Worker\nModel Runner", PURPLE)
    box(ax, 30, 650, 235, 110, "结果交付", "token → 文本\n关联请求 · 聚合／流式", BLUE)
    group(ax, 650, 650, 520, 125, "设备侧的数据与计算", PURPLE, "#FCFAFF")
    box(ax, 675, 705, 220, 50, "权重 · 前向 · 采样", color=PURPLE)
    box(ax, 935, 705, 205, 50, "KV 张量", color=GREEN)
    arrow(ax, [(265, 220), (355, 220)], "请求", color=BLUE)
    arrow(ax, [(405, 270), (405, 390)], "调度／更新", (405, 320), ORANGE)
    arrow(ax, [(515, 390), (515, 270)], "计划／状态", (515, 355), ORANGE)
    arrow(ax, [(560, 430), (620, 430)], "申请／释放", (590, 400), GREEN, size=12)
    arrow(ax, [(620, 480), (560, 480)], "分配结果", (590, 525), GREEN, size=12)
    arrow(ax, [(835, 205), (925, 205)], "计划", color=PURPLE)
    arrow(ax, [(925, 260), (835, 260)], "结果", (880, 288), PURPLE)
    arrow(ax, [(355, 250), (290, 250), (290, 610), (147, 610), (147, 650)],
          "token／完成信息", (170, 585), BLUE)
    arrow(ax, [(1050, 295), (1050, 650)], "执行与读写", (1050, 480), PURPLE)
    arrow(ax, [(730, 510), (730, 600), (1035, 600), (1035, 705)],
          "位置映射约束（经执行层使用）", (855, 600), GREEN, dashed=True, size=13)
    note(ax, 815, "绿色管理框保存管理信息；设备框中的 KV 才是被模型读写的张量。")
    save(fig, "fig-1-2-components")


def sequence(title, subtitle, actors, height=800, xs=None):
    fig, ax = canvas(title, subtitle, height)
    xs = xs or [130 + i * (940 / (len(actors) - 1)) for i in range(len(actors))]
    for x, (name, color) in zip(xs, actors):
        box(ax, x - 110, 125, 220, 65, name, color=color)
        ax.plot([x, x], [190, height - 75], color="#BCC6D2", linewidth=1.2, linestyle="--", zorder=0)
    return fig, ax, xs


def msg(ax, xs, a, b, y, label, color=MUTED, dashed=False):
    arrow(ax, [(xs[a], y), (xs[b], y)], label, color=color, dashed=dashed)


def iteration():
    fig, ax, xs = sequence("图 1.3  一轮计算的协作闭环", "实线：调用或提交；虚线：返回。按逻辑先后展示，不表示实际耗时或进程归属。", [
        ("EngineCore", ORANGE), ("Scheduler", ORANGE),
        ("KVCacheManager", GREEN), ("执行层", PURPLE)], 1040,
        xs=[150, 450, 750, 1050])
    text(ax, 40, 225, "前端登记 A、B → 核心接纳请求，Scheduler 保存请求状态", 15, BLUE, ha="left", bg=True)
    group(ax, 25, 265, 1150, 605, "逻辑迭代：未结束则继续（实际实现可重叠部分工作）", MUTED, "#FAFBFD", True)
    msg(ax, xs, 0, 1, 330, "① 安排本轮工作", ORANGE)
    msg(ax, xs, 1, 2, 390, "② 申请所需 KV 位置", GREEN)
    msg(ax, xs, 2, 1, 445, "分配结果／块位置", GREEN, True)
    msg(ax, xs, 1, 0, 505, "③ 返回本轮计划", ORANGE, True)
    msg(ax, xs, 0, 3, 565, "④ 提交计划：请求工作量与 KV 位置等", PURPLE)
    msg(ax, xs, 3, 0, 625, "⑤ 返回计算结果：新 token 等", PURPLE, True)
    msg(ax, xs, 0, 1, 685, "⑥ 据结果更新请求", ORANGE)
    text(ax, xs[1], 730, "更新进度 · 判断结束", 14, ORANGE, bg=True)
    msg(ax, xs, 1, 2, 785, "结束时按安全条件释放引用", GREEN)
    msg(ax, xs, 1, 0, 840, "⑦ 返回 token／完成信息", ORANGE, True)
    text(ax, 40, 920, "EngineCore → 前端：新 token／完成信息；前端整理并按约定交付", 15, BLUE, ha="left", bg=True)
    note(ax, 990, "EngineCore 组织交接；Scheduler 决定工作与更新请求；KV 管理器安排位置；执行层落实计算。")
    save(fig, "fig-1-3-iteration")


def deployment():
    fig, ax = canvas("图 1.4  相同职责的两种进程放置方式", "两边都限定为单设备 UniProcExecutor；比较核心多进程关闭与开启", 830)
    text(ax, 310, 130, "同进程教学形态", 19, INK, weight="bold")
    text(ax, 890, 130, "前端与核心分进程", 19, INK, weight="bold")
    group(ax, 55, 170, 510, 450, "进程 P：CPU 上的 Python 程序")
    box(ax, 100, 230, 420, 75, "前端：输入适配与结果处理", color=BLUE)
    box(ax, 100, 360, 420, 75, "EngineCore：调度与 KV 管理", color=ORANGE)
    box(ax, 100, 490, 420, 75, "执行：Worker／Model Runner", color=PURPLE)
    arrow(ax, [(310, 305), (310, 360)], "进程内调用", (405, 330))
    arrow(ax, [(310, 435), (310, 490)], color=PURPLE)
    group(ax, 635, 170, 510, 155, "进程 P：前端（CPU）")
    box(ax, 680, 230, 420, 65, "输入适配与结果处理", color=BLUE)
    group(ax, 635, 405, 510, 215, "进程 Q：核心与执行（CPU）")
    box(ax, 680, 455, 420, 60, "EngineCore：调度与 KV 管理", color=ORANGE)
    box(ax, 680, 540, 420, 55, "Worker／Model Runner", color=PURPLE)
    arrow(ax, [(840, 325), (840, 405)], "请求", (780, 365), BLUE)
    arrow(ax, [(950, 405), (950, 325)], "结果", (1015, 365), BLUE)
    text(ax, 890, 365, "IPC", 15, MUTED, bg=True)
    arrow(ax, [(890, 515), (890, 540)], color=PURPLE)
    for center in [310, 890]:
        box(ax, center - 225, 690, 450, 70, "设备：模型计算 · 权重 · KV 张量", color=PURPLE)
        arrow(ax, [(center, 620), (center, 690)], "提交设备工作", (center + 95, 655), PURPLE)
    note(ax, 800, "角色数 ≠ 进程数；图中 Worker 控制代码运行于 CPU，设备执行提交的算子。")
    save(fig, "fig-1-4-deployment")


def focus():
    fig, ax = canvas("图 2.1  本章在全景中的位置", "蓝色是本章重点：输入适配与结果交付；核心计算暂时折叠", 510)
    box(ax, 35, 215, 210, 105, "调用者", "离线程序／在线客户端", MUTED)
    box(ax, 365, 145, 300, 100, "输入适配", "整理输入 · 校验 · 登记", BLUE)
    box(ax, 365, 345, 300, 100, "结果交付", "关联请求 · 文本 · 结束", BLUE)
    box(ax, 860, 240, 300, 115, "核心与执行", "调度 · KV · 模型计算\n内部职责见图 1.2", MUTED)
    arrow(ax, [(245, 245), (295, 245), (295, 195), (365, 195)], "需求", (300, 175), BLUE)
    arrow(ax, [(665, 195), (1010, 195), (1010, 240)], "计算请求", (810, 175), BLUE)
    arrow(ax, [(1010, 355), (1010, 395), (665, 395)], "token／完成信息", (825, 420), BLUE)
    arrow(ax, [(365, 395), (140, 395), (140, 320)], "最终结果或响应片段", (230, 420), BLUE)
    save(fig, "fig-2-1-focus")


def representations():
    fig, ax = canvas("图 2.2  一条请求在不同边界上的表示", "上排去程向右，下排回程向左；A 的身份贯穿往返，编号仅为教学构造", 660)
    box(ax, 40, 170, 285, 120, "用户需求 A", "一段文本\n最多生成 2 个 token", BLUE)
    box(ax, 455, 170, 285, 120, "适配后的输入", "token：[11, 27, 42]\n保留生成要求", BLUE)
    box(ax, 870, 170, 285, 120, "核心计算请求", "身份 A · token 输入\n生成约束", ORANGE)
    box(ax, 870, 450, 285, 120, "核心内部结果", "身份 A · 新 token\n进度／完成信息", ORANGE)
    box(ax, 455, 450, 285, 120, "输出处理状态", "关联 A · 累积 token\n恢复文本 · 检查停止", BLUE)
    box(ax, 40, 450, 285, 120, "用户结果 A", "生成文本与结果信息\n按交付方式返回", BLUE)
    arrow(ax, [(325, 230), (455, 230)], "输入转换", color=BLUE)
    arrow(ax, [(740, 230), (870, 230)], "登记／提交", color=ORANGE)
    arrow(ax, [(1012, 290), (1012, 450)], "调度与执行", (1012, 375), PURPLE)
    arrow(ax, [(870, 510), (740, 510)], "按身份归并", color=BLUE)
    arrow(ax, [(455, 510), (325, 510)], "交付", color=BLUE)
    note(ax, 620, "一份逻辑请求可由不同对象表示；这些框不是完整字段表，也不是六个进程。")
    save(fig, "fig-2-2-representations")


def offline():
    fig, ax, xs = sequence("图 2.3  离线调用的计算进度与交付时刻", "A 生成 2 个 token，B 生成 3 个；只画逻辑顺序，不画实际耗时", [
        ("用户程序", MUTED), ("离线前端", BLUE), ("核心与执行", ORANGE)], 920)
    msg(ax, xs, 0, 1, 240, "提交 A、B 并等待", BLUE)
    msg(ax, xs, 1, 2, 310, "登记两条计算请求", BLUE)
    msg(ax, xs, 2, 1, 400, "第 1 步：A、B 各产生首 token", PURPLE, True)
    msg(ax, xs, 2, 1, 500, "第 2 步：各增一 token，A 完成", PURPLE, True)
    text(ax, xs[1], 550, "保存 A 的最终结果", 16, BLUE, bg=True)
    msg(ax, xs, 2, 1, 645, "第 3 步：B 再增一 token，完成", PURPLE, True)
    text(ax, xs[1], 695, "保存 B；整理整组结果", 16, BLUE, bg=True)
    msg(ax, xs, 1, 0, 780, "返回 [A 的结果，B 的结果]", BLUE)
    ax.add_patch(Rectangle((xs[0] - 6, 248), 12, 523, facecolor="#E4E8ED", edgecolor="none", zorder=0))
    text(ax, 260, 525, "本次调用仍在等待\n内部已有多次进展", 15, MUTED, bg=True)
    note(ax, 875, "FINAL_ONLY 约束用户输出，不表示核心在结束前不向前端传递内部结果。")
    save(fig, "fig-2-3-offline")


def online():
    fig, ax = canvas("图 2.4  离线入口与在线服务的责任对照", "共同的是核心计算职责；两条示意路径不表示共享同一个运行实例", 740)
    group(ax, 25, 135, 790, 200, "离线：Python 调用与整组返回", BLUE)
    box(ax, 55, 205, 230, 85, "Python 程序", "提交提示词列表", MUTED)
    box(ax, 445, 205, 330, 85, "离线前端", "适配输入 · 收集最终结果", BLUE)
    arrow(ax, [(285, 230), (445, 230)], "调用", color=BLUE)
    arrow(ax, [(445, 275), (285, 275)], "整组结果", (365, 305), BLUE)
    group(ax, 25, 405, 790, 235, "在线：网络协议、独立请求、异步等待", BLUE)
    box(ax, 55, 480, 180, 100, "客户端", "各自提交／接收", MUTED)
    box(ax, 315, 480, 200, 100, "HTTP 协议层", "解析请求\n组织响应", BLUE)
    box(ax, 590, 480, 190, 100, "异步前端", "适配／关联\n等待内部结果", BLUE)
    arrow(ax, [(235, 505), (315, 505)], "请求", color=BLUE)
    arrow(ax, [(315, 560), (235, 560)], "响应", (275, 595), BLUE)
    arrow(ax, [(515, 505), (590, 505)], color=BLUE)
    arrow(ax, [(590, 560), (515, 560)], color=BLUE)
    box(ax, 930, 310, 245, 170, "同类核心职责", "调度请求\n管理 KV\n组织设备执行", ORANGE)
    arrow(ax, [(775, 230), (1040, 230), (1040, 310)], "计算请求", (920, 210), ORANGE)
    arrow(ax, [(780, 505), (1040, 505), (1040, 480)], color=ORANGE)
    arrow(ax, [(930, 365), (855, 365), (855, 275), (775, 275)], color=BLUE, dashed=True)
    arrow(ax, [(930, 425), (865, 425), (865, 560), (780, 560)], color=BLUE, dashed=True)
    note(ax, 685, "实线：请求交接；虚线：内部结果。在线响应可选择流式，也可一次性返回完整结果。")
    save(fig, "fig-2-4-online")


def cancel():
    fig, ax, xs = sequence("图 2.5  取消沿控制路径传播", "停止等待，需要传播为计算生命周期事件；已经提交的设备工作可能仍在完成", [
        ("客户端／调用任务", MUTED), ("在线前端", BLUE), ("核心／调度与 KV", ORANGE)], 820)
    msg(ax, xs, 0, 1, 260, "离开／取消生成任务", BLUE)
    msg(ax, xs, 1, 2, 350, "终止请求 B", ORANGE)
    text(ax, xs[2], 440, "结束请求的活跃状态\n协调安全清理", 16, ORANGE, bg=True)
    msg(ax, xs, 2, 1, 550, "可能晚到的内部结果", MUTED, True)
    text(ax, xs[1], 610, "忽略已取消请求的结果\n清理交付侧状态", 16, BLUE, bg=True)
    note(ax, 725, "解除 KV 引用 ≠ 必须擦除缓存内容；清理还需尊重在途执行与存储安全。")
    note(ax, 770, "本图不表示网络断开会立即被检测，也不表示执行算子可以被瞬间撤销。")
    save(fig, "fig-2-5-cancel")


def main():
    for draw in [boundary, components, iteration, deployment, focus, representations, offline, online, cancel]:
        draw()


if __name__ == "__main__":
    main()
