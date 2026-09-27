"""
InsightFace (buffalo_l) 人脸相似度工具：1:1 与 1:N，支持命令行，也支持直接双击。

只依赖 insightface_face_similarity.py 已有的公开接口，不修改它。
example_insightface.py 那个最简示例保持原样，这个是完整版。

关于数值：ArcFace 512 维向量的余弦相似度，范围 -100 ~ 100。
同一个人通常是 45 ~ 70，不同人通常是 0 ~ 40。
这个刻度和 dlib 版的百分比不能互相比较，阈值要拿自己的样本标定。

命令行
    python example2_insightface.py a.jpg b.jpg
        1:1，两张图比一次。

    python example2_insightface.py 查询这是谁.jpg 1用户1.jpg 2用户2.jpg 3帅哥3.jpg
        1:N，一张查询图逐个和后面所有候选比，按相似度排名。

    python example2_insightface.py 查询这是谁.jpg face_library -n 5
        1:N，候选可以直接给文件夹，递归收集里面的图片。

    python example2_insightface.py 查询这是谁.jpg face_library --json
        输出 JSON。

双击
    双击本文件（不带任何参数运行）会进入交互模式，按提示输入路径即可，
    结束后窗口会停住等回车，不会一闪而过。

在代码里用
    from example2_insightface import compare_two, search_candidates, format_report

    score = compare_two("a.jpg", "b.jpg")

    report = search_candidates("查询这是谁.jpg", ["1用户1.jpg", "2用户2.jpg"])
    print(format_report(report))
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, Sequence

from insightface_face_similarity import (
    MODEL_NAME,
    ImageInput,
    best_pair_percent,
    encode_faces,
    warm_up,
)

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BAR_WIDTH = 10
LINE_WIDTH = 78

# ArcFace 一张图大约 0.1 ~ 0.3 秒，候选多的时候给个进度提示。
PROGRESS_THRESHOLD = 20


# -------------------- Text Layout --------------------
def display_width(text: str) -> int:
    """Terminal columns a string occupies, counting CJK as two."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def pad(text: str, width: int) -> str:
    """Left align text to a column width that survives CJK filenames."""
    return text + " " * max(0, width - display_width(text))


def safe_char(char: str, fallback: str) -> str:
    """Fall back to ASCII when the console encoding cannot show a glyph."""
    encoding = sys.stdout.encoding or "ascii"
    try:
        char.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return fallback
    return char


def bar(value: float, low: float = 0.0, high: float = 100.0, width: int = BAR_WIDTH) -> str:
    """A gauge of exactly BAR_WIDTH cells, filled between low and high.

    Scores in one search are often within a point of each other, and a plain
    0-100 bar would then look identical for all of them. Callers pass the real
    window they used so the printed range can be shown next to the table.
    """
    span = high - low
    ratio = 0.0 if span <= 0 else (value - low) / span
    filled = max(0, min(width, round(ratio * width)))
    full = safe_char("█", "#")
    empty = safe_char("░", ".")
    return full * filled + empty * (width - filled)


def shorten(text: str, width: int) -> str:
    """Middle-trim a name so a long path cannot push the table off screen."""
    if display_width(text) <= width:
        return text
    ellipsis = safe_char("…", "...")
    keep = max(4, width - display_width(ellipsis) - 1)
    head = keep // 2
    tail = keep - head
    return text[:head] + ellipsis + (text[-tail:] if tail else "")


def rule(title: str = "", width: int = LINE_WIDTH) -> str:
    """A section heading so the output is readable when it is long."""
    line = safe_char("─", "-") * width
    if not title:
        return line
    return f"{title}\n{line}"


def back_to_menu() -> None:
    """Wait for Enter, then let the interactive loop show the menu again.

    The line is always consumed, even when stdin is a pipe, so a scripted
    session stays in step with the prompts.
    """
    try:
        input(safe_char("\n按回车回到主菜单…", "\nPress Enter for the menu..."))
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(0)


def pause() -> None:
    """Hold the window open so a double-clicked run can be read."""
    try:
        input(safe_char("\n按回车键关闭…", "\nPress Enter to close..."))
    except (EOFError, KeyboardInterrupt):
        pass


def progress_line(done: int, total: int) -> None:
    """Overwrite one terminal line, so long searches are not silent."""
    if not sys.stderr.isatty():
        return
    sys.stderr.write(f"\r  正在提取特征 {done}/{total} …")
    sys.stderr.flush()


def progress_done() -> None:
    if sys.stderr.isatty():
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()


# -------------------- Path Handling --------------------
def resolve_path(text: str) -> Optional[Path]:
    """Accept a pasted path, strip quotes, and fall back to the script folder."""
    cleaned = text.strip().strip('"').strip("'")
    if not cleaned:
        return None
    cleaned = os.path.expandvars(os.path.expanduser(cleaned))
    for candidate in (Path(cleaned), SCRIPT_DIRECTORY / cleaned):
        if candidate.exists():
            return candidate
    return Path(cleaned)


def expand_candidates(items: Iterable, recursive: bool = True) -> list:
    """Turn files and folders into a flat, sorted, de-duplicated image list."""
    found: list = []
    missing: list = []

    for item in items:
        path = item if isinstance(item, Path) else resolve_path(str(item))
        if path is None:
            continue
        if path.is_dir():
            walker = path.rglob("*") if recursive else path.glob("*")
            for child in sorted(walker):
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                    found.append(child)
        elif path.is_file():
            found.append(path)
        else:
            missing.append(str(item))

    if missing:
        print("找不到，已跳过: " + "、".join(missing), file=sys.stderr)

    unique: list = []
    seen: set = set()
    for path in found:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


# -------------------- Scoring --------------------
def compare_two(
    image_a: ImageInput,
    image_b: ImageInput,
    largest_only: bool = False,
) -> Optional[float]:
    """1:1 similarity as a percentage, or None when a face is missing."""
    embeddings_a = encode_faces(image_a)
    embeddings_b = encode_faces(image_b)
    if not embeddings_a or not embeddings_b:
        return None
    return best_pair_percent(embeddings_a, embeddings_b, largest_only)


def search_candidates(
    query_image: ImageInput,
    candidates: Sequence,
    top: Optional[int] = None,
    largest_only: bool = False,
    show_progress: bool = True,
) -> dict:
    """1:N search. Returns a dict so the caller can format it however it likes.

    Keys: query, query_faces, matches, skipped, elapsed_seconds.
    Each match is {"rank", "candidate", "similarity", "faces"}.
    A candidate that cannot be read or holds no face lands in skipped
    instead of aborting the whole search.
    """
    started = time.perf_counter()
    query_embeddings = encode_faces(query_image)
    if not query_embeddings:
        raise ValueError("查询图里没有检测到人脸")

    verbose = show_progress and len(candidates) >= PROGRESS_THRESHOLD
    if verbose:
        print(f"共 {len(candidates)} 张候选，正在提取特征，首次约 2 秒加载模型…")

    matches: list = []
    skipped: list = []
    for index, candidate in enumerate(candidates, start=1):
        if verbose:
            progress_line(index, len(candidates))
        try:
            candidate_embeddings = encode_faces(candidate)
        except (OSError, ValueError) as error:
            skipped.append({"candidate": str(candidate), "reason": f"读取失败: {error}"})
            continue
        if not candidate_embeddings:
            skipped.append({"candidate": str(candidate), "reason": "未检测到人脸"})
            continue
        matches.append(
            {
                "candidate": str(candidate),
                "similarity": best_pair_percent(
                    query_embeddings, candidate_embeddings, largest_only
                ),
                "faces": len(candidate_embeddings),
            }
        )

    if verbose:
        progress_done()

    matches.sort(key=lambda row: row["similarity"], reverse=True)
    if top is not None and top > 0:
        matches = matches[:top]
    for rank, row in enumerate(matches, start=1):
        row["rank"] = rank

    return {
        "query": str(query_image),
        "query_faces": len(query_embeddings),
        "model": MODEL_NAME,
        "matches": matches,
        "skipped": skipped,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


# -------------------- Output --------------------
def format_pair(first_name: str, second_name: str, score: float, show_bar: bool = True) -> str:
    """Render a 1:1 result."""
    lines = [
        f"图 A: {first_name}",
        f"图 B: {second_name}",
        f"相似度: {score:.2f}%",
    ]
    if show_bar:
        lines.append("分布:   " + bar(score, 0.0, 100.0))
    return "\n".join(lines)


def format_report(report: dict, show_bar: bool = True) -> str:
    """Render a search_candidates() report as an aligned text table."""
    lines = [
        f"查询图: {report['query']}  ({report['query_faces']} 张脸)",
        (
            f"候选: {len(report['matches'])} 张可比对"
            f"，跳过 {len(report['skipped'])} 张"
            f"，用时 {report['elapsed_seconds']} s"
        ),
    ]

    matches = report["matches"]
    if matches:
        prefix = "  排名  相似度    脸数  "
        gauge_width = BAR_WIDTH if show_bar else 0
        budget = LINE_WIDTH - display_width(prefix) - gauge_width - 2
        name_width = max(12, min(
            max(display_width(row["candidate"]) for row in matches), budget
        ))
        scores = [row["similarity"] for row in matches]
        low, high = min(scores), max(scores)
        if len(matches) == 1 or high - low < 0.5:
            low, high = 0.0, 100.0
            scaled = False
        else:
            scaled = True

        header = prefix + pad("候选", name_width)
        if show_bar:
            header += "  " + pad("分布", gauge_width)
        lines.append("")
        lines.append(header)
        lines.append("  " + safe_char("─", "-") * (display_width(header) - 2))
        for row in matches:
            line = (
                f"  {row['rank']:>4}  {row['similarity']:>6.2f}%"
                f"  {row['faces']:>4}  "
                + pad(shorten(Path(row["candidate"]).name, name_width), name_width)
            )
            if show_bar:
                line += "  " + bar(row["similarity"], low, high, gauge_width)
            lines.append(line)
        best = matches[0]
        lines.append("")
        if show_bar and scaled:
            lines.append(
                f"分布条按本次 {low:.2f}% - {high:.2f}% 缩放，"
                "便于看清接近的分数；左边数值是真实相似度。"
            )
        lines.append(f"最像的是 {Path(best['candidate']).name} ({best['similarity']:.2f}%)")
        lines.append(
            "参考: 同一个人约 45~70，不同人约 0~40，"
            "阈值请用自己的样本标定。"
        )

    if report["skipped"]:
        lines.append("")
        for row in report["skipped"]:
            lines.append(f"跳过 {Path(row['candidate']).name}: {row['reason']}")

    return "\n".join(lines)


# -------------------- Interactive (double click) --------------------
def ask(prompt: str) -> str:
    """One prompt that behaves the same on every console."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)


def ask_existing(prompt: str) -> Path:
    """Keep asking until the user gives a path that exists."""
    while True:
        typed = ask(prompt)
        if not typed:
            return None
        path = resolve_path(typed)
        if path is not None and path.exists():
            return path
        print("  这个路径不存在，请检查后重试，或直接回车返回。")


def run_interactive() -> int:
    """Menu driven flow, used when the file is started without arguments."""
    print(rule("InsightFace 人脸相似度  " + safe_char("█", "#") + "  1:1 / 1:N"))
    print(f"内核: {MODEL_NAME} · 512 维 ArcFace · onnxruntime")
    print("当前目录: " + str(Path.cwd()))
    print("脚本目录: " + str(SCRIPT_DIRECTORY))
    print("提示: 路径可以直接粘贴，回车确认；候选填文件夹会自动递归收图。")
    print("提示: 第一次比对要先加载模型，约 2 秒，之后每张图 0.1~0.3 秒。")

    while True:
        print()
        print(rule("请选择模式"))
        print("  1) 1:1   两张图互相比一次")
        print("  2) 1:N   一张查询图，逐个和多个候选比，输出排名")
        print("  0) 退出")
        choice = ask("请输入编号 [2]: ") or "2"

        if choice == "0":
            print("再见。")
            return 0

        if choice == "1":
            first = ask_existing("图 A 路径: ")
            if first is None:
                continue
            second = ask_existing("图 B 路径: ")
            if second is None:
                continue
            started = time.perf_counter()
            try:
                score = compare_two(first, second)
            except (OSError, ValueError) as error:
                print(f"出错了: {error}")
                continue
            print()
            if score is None:
                print("有一张图没有检测到人脸。")
            else:
                print(format_pair(first.name, second.name, score))
            print(f"用时: {time.perf_counter() - started:.3f} s")
            back_to_menu()
            continue

        if choice != "2":
            print("请输入 0、1 或 2。")
            continue

        query = ask_existing("查询图路径（单张）: ")
        if query is None:
            continue

        typed: list = []
        print("现在逐个输入候选，可以是图片，也可以是整个文件夹。")
        while True:
            entry = ask("候选路径（直接回车结束）: ")
            if not entry:
                break
            typed.append(entry)

        candidates = expand_candidates(typed)
        if not candidates:
            print("没有可用的候选图片。")
            back_to_menu()
            continue

        top_text = ask(f"最多显示前几条？候选共 {len(candidates)} 张 [直接回车=全部]: ")
        top = None
        if top_text:
            try:
                # 0 or less means "all", same as -n 0 on the command line.
                top = int(top_text)
                top = top if top > 0 else None
            except ValueError:
                print("  不是数字，将显示全部。")

        try:
            report = search_candidates(query, candidates, top=top)
        except (OSError, ValueError) as error:
            print(f"出错了: {error}")
            back_to_menu()
            continue

        print()
        print(format_report(report))
        back_to_menu()


# -------------------- Command Line --------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InsightFace 人脸相似度：1:1 比两张图，1:N 用一张查询图搜一批候选。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python example2_insightface.py a.jpg b.jpg\n"
            "  python example2_insightface.py 查询这是谁.jpg 1用户1.jpg 2用户2.jpg 3帅哥3.jpg\n"
            "  python example2_insightface.py 查询这是谁.jpg face_library -n 5 --json\n"
            "\n"
            "不带任何参数运行会进入交互模式，也可以直接双击本文件。\n"
            "数值为 512 维 ArcFace 余弦相似度百分比：同一个人约 45~70，不同人约 0~40。"
        ),
    )
    parser.add_argument("query", nargs="?", help="查询图")
    parser.add_argument("candidates", nargs="*", help="候选图或候选文件夹")
    parser.add_argument(
        "-n", "--top", type=int, default=0, help="最多显示前 N 条，0 表示全部"
    )
    parser.add_argument(
        "--largest-only",
        action="store_true",
        help="只比较每张图里最大的一张脸，而不是任意配对",
    )
    parser.add_argument(
        "--no-recursive", action="store_true", help="候选文件夹只看第一层"
    )
    parser.add_argument("--no-bar", action="store_true", help="不画相似度条")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser


def run_from_arguments(arguments: argparse.Namespace) -> int:
    if not arguments.candidates:
        print("至少要给一个候选，文件夹也可以。", file=sys.stderr)
        return 2

    query = resolve_path(arguments.query)
    if query is None or not query.is_file():
        print(f"查询图不存在: {arguments.query}", file=sys.stderr)
        return 2

    candidates = expand_candidates(
        arguments.candidates, recursive=not arguments.no_recursive
    )
    if not candidates:
        print("没有可用的候选图片", file=sys.stderr)
        return 2

    try:
        if len(candidates) == 1 and not arguments.json:
            started = time.perf_counter()
            score = compare_two(
                query, candidates[0], largest_only=arguments.largest_only
            )
            if score is None:
                print("有一张图没有检测到人脸")
                return 1
            print(format_pair(query.name, candidates[0].name, score,
                              show_bar=not arguments.no_bar))
            print(f"用时: {time.perf_counter() - started:.3f} s")
            return 0

        report = search_candidates(
            query,
            candidates,
            top=arguments.top or None,
            largest_only=arguments.largest_only,
        )
    except (OSError, ValueError) as error:
        print(f"出错了: {error}", file=sys.stderr)
        return 1

    if arguments.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report, show_bar=not arguments.no_bar))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] in ("-h", "--help", "info"):
        build_parser().print_help()
        return 0

    if not argv:
        print(f"正在加载 {MODEL_NAME} 模型，约 2 秒…")
        warm_up()
        status = run_interactive()
        if status == 0:
            pause()
        return status

    arguments = build_parser().parse_args(argv)
    if arguments.query is None:
        build_parser().print_help()
        return 2
    return run_from_arguments(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
