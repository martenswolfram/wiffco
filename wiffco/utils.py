from typing import List

def print_table(list_of_row_lists: List[List[str]],
                col_sep: str = " | ",
                row_sep: str = "-"):
    if not len(list_of_row_lists):
        return "<empty table>"
    num_rows = len(list_of_row_lists)
    num_cols = len(list_of_row_lists[0])
    # First sweep to determine width (no. of characters) and height (no. of lines) per cell
    raw_cell_line_collection = []
    cell_heights = [0] * num_rows
    cell_widths = [0] * num_cols
    for i, row_list in enumerate(list_of_row_lists):
        raw_cell_line_collection.append([])
        if len(row_list) != num_cols:
            raise ValueError("Inconsistent table dimensions.")
        for j, cell_str in enumerate(row_list):
            cell_lines = cell_str.splitlines()
            pass
            cell_width = max(len(line) for line in cell_lines)
            cell_height = len(cell_lines)
            cell_widths[j] = max(cell_widths[j], cell_width)
            cell_heights[i] = max(cell_heights[i], cell_height)
            raw_cell_line_collection[-1].append(cell_lines)
    
    full_width = sum(cell_widths) + (num_cols - 1) * len(col_sep)
    # Second sweep to generate content
    rows = []
    for i, row_list in enumerate(list_of_row_lists):
        for row_line in range(cell_heights[i]):
            row = []
            for j, row_cell_lines in enumerate(raw_cell_line_collection[i]):
                width = cell_widths[j]
                try:
                    row.append(row_cell_lines[row_line].rjust(width))
                except IndexError:
                    row.append(" " * width)
            rows.append(col_sep.join(row))
        rows.append(row_sep * full_width)
    return "\n".join(rows) + "\n"
