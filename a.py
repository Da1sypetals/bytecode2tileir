#!/usr/bin/env python3
"""
网格打印脚本
值计算公式: value = 20*i + j
i 为行索引，j 为列索引
"""


def print_grid(rows, cols, cell_width=4):
    """
    打印格式化的网格

    Args:
        rows: 行数
        cols: 列数
        cell_width: 每个单元格的宽度
    """
    # 打印表头
    header = "     " + " ".join(f"{j:^{cell_width}}" for j in range(cols))
    print(header)
    print("    +" + "+".join("-" * cell_width for _ in range(cols)) + "+")

    # 打印每一行
    for i in range(rows):
        values = [20 * i + j for j in range(cols)]
        row_str = f"{i:>3} |" + "|".join(f"{v:^{cell_width}}" for v in values) + "|"
        print(row_str)
        print("    +" + "+".join("-" * cell_width for _ in range(cols)) + "+")


if __name__ == "__main__":
    print_grid(rows=8, cols=8, cell_width=5)
