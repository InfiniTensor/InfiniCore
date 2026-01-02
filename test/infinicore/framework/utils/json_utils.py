import json
import os
from datetime import datetime


def save_json_report(save_path, total_results):
    """
    Saves the report list to a JSON file with specific custom formatting
    (Compact for short lines, Expanded for long lines).
    """
    directory, filename = os.path.split(save_path)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    final_path = os.path.join(directory, f"{name}_{timestamp}{ext}")

    # Define Indentation
    I4, I8, I12, I16, I20 = " " * 4, " " * 8, " " * 12, " " * 16, " " * 20

    print(f"💾 Saving to: {final_path}")

    # Helper for JSON stringify to avoid repetition
    def _to_json(obj):
        return json.dumps(obj, ensure_ascii=False)

    try:
        with open(final_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            for i, entry in enumerate(total_results):
                f.write(f"{I4}{{\n")
                keys = list(entry.keys())

                for j, key in enumerate(keys):
                    val = entry[key]
                    comma = "," if j < len(keys) - 1 else ""

                    # Special handling for 'testcases' list
                    if key == "testcases" and isinstance(val, list):
                        f.write(f'{I8}"{key}": [\n')
                        for c_idx, case_item in enumerate(val):
                            f.write(f"{I12}{{\n")
                            case_keys = list(case_item.keys())

                            # Filter out keys that we handle specially at the end
                            standard_keys = [
                                k
                                for k in case_keys
                                if k not in ["comparison_target", "tolerance"]
                            ]

                            for k_idx, c_key in enumerate(standard_keys):
                                c_val = case_item[c_key]
                                # Determine comma logic
                                c_comma = (
                                    ","
                                    if k_idx < len(standard_keys) - 1
                                    or "comparison_target" in case_item
                                    else ""
                                )

                                if c_key in ["kwargs", "inputs"]:
                                    _write_field(
                                        f, c_key, c_val, I16, I20, close_comma=c_comma
                                    )
                                else:
                                    f.write(
                                        f'{I16}"{c_key}": {_to_json(c_val)}{c_comma}\n'
                                    )

                            # Handle trailing comparison/tolerance fields uniformly
                            if "comparison_target" in case_item:
                                cmp = _to_json(case_item.get("comparison_target"))
                                tol = _to_json(case_item.get("tolerance"))
                                f.write(
                                    f'{I16}"comparison_target": {cmp}, "tolerance": {tol}\n'
                                )

                            close_case = "," if c_idx < len(val) - 1 else ""
                            f.write(f"{I12}}}{close_case}\n")
                        f.write(f"{I8}]{comma}\n")
                    else:
                        # Standard top-level fields
                        f.write(f"{I8}{_to_json(key)}: {_to_json(val)}{comma}\n")

                close_entry = "}," if i < len(total_results) - 1 else "}"
                f.write(f"{I4}{close_entry}\n")
            f.write("]\n")
        print(f"   ✅ Saved.")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"   ❌ Save failed: {e}")


def _write_field(f, key, value, indent, sub_indent, close_comma=""):
    """
    Internal Helper: Write a JSON field with wrapping.
    """
    # 1. Try Compact Mode
    compact_json = json.dumps(value, ensure_ascii=False)
    if len(compact_json) <= 180:
        f.write(f'{indent}"{key}": {compact_json}{close_comma}\n')
        return

    # 2. Fill/Flow Mode
    is_dict = isinstance(value, dict)
    open_char = "{" if is_dict else "["
    close_char = "}" if is_dict else "]"

    f.write(f'{indent}"{key}": {open_char}')

    if is_dict:
        items = list(value.items())
    else:
        items = value

    current_len = len(indent) + len(f'"{key}": {open_char}')

    for i, item in enumerate(items):
        if is_dict:
            k, v = item
            val_str = json.dumps(v, ensure_ascii=False)
            item_str = f'"{k}": {val_str}'
        else:
            item_str = json.dumps(item, ensure_ascii=False)

        is_last = i == len(items) - 1
        item_comma = "" if is_last else ", "

        if current_len + len(item_str) + len(item_comma) > 180:
            f.write(f"\n{sub_indent}")
            current_len = len(sub_indent)

        f.write(f"{item_str}{item_comma}")
        current_len += len(item_str) + len(item_comma)

    f.write(f"{close_char}{close_comma}\n")
