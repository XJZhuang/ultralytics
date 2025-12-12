#!/usr/bin/env python3
"""
YAML -> Mermaid Model Structure Graph (Backbone / Neck / Head 三列)

Usage:
    python yml2modelgraph_mermaid.py model.yaml [out_name]

Outputs:
    out_name.mmd (Mermaid file, 可直接丢到 mermaid.live / VSCode)
"""

import yaml
import sys
import os
import math
from typing import List, Any, Dict, Tuple

# ---------------- CONFIG ----------------
# 可选字段: "name", "channels", "n", "args"
SHOW_FIELDS = ["name"]


# ---------------- utils ----------------
def make_divisible(v, divisor=8):
    if v is None:
        return None
    return int(math.ceil(float(v) / divisor) * divisor)


def safe_eval(x):
    if not isinstance(x, str):
        return x
    try:
        return eval(x)
    except Exception:
        return x


def module_name_lower(m_raw):
    if isinstance(m_raw, str):
        return m_raw.lower()
    try:
        return m_raw.__name__.lower()
    except Exception:
        return str(m_raw).lower()


# ---------------- parse model ----------------
def parse_yaml_model(d: Dict[str, Any], input_channels=3):
    gd = float(d.get("depth_multiple", 1.0))
    gw = float(d.get("width_multiple", 1.0))
    seq_backbone = d.get("backbone", []) or []
    seq_head = d.get("head", []) or []
    seq = seq_backbone + seq_head

    ch = [input_channels]
    layers, edges = [], []

    for i, entry in enumerate(seq):
        if not (isinstance(entry, list) and len(entry) >= 4):
            continue
        f, n, m, args = entry
        from_idxs = [f] if isinstance(f, int) else list(f) if hasattr(f, "__iter__") else [f]
        try:
            n = int(n)
        except:
            n = 1
        args = [safe_eval(a) for a in (args or [])]
        group = "backbone" if i < len(seq_backbone) else "head"
        mname = module_name_lower(m)

        # --- 输入通道 ---
        c1s = []
        for src in from_idxs:
            try:
                src_i = int(src)
            except:
                src_i = -1
            if src_i == -1:
                c1s.append(ch[-1])
            else:
                mapped = src_i + 1
                c1s.append(ch[mapped] if 0 <= mapped < len(ch) else ch[-1])

        # --- 输出通道 ---
        if "concat" in mname:
            c2 = sum(c1s)
        else:
            c2 = args[0] if args and isinstance(args[0], int) else (c1s[0] if c1s else 0)
            if not ("detect" in mname):
                try:
                    c2 = make_divisible(c2 * gw)
                except:
                    pass
        n_ = max(round(n * gd), 1) if n > 1 else n

        # --- label内容配置 ---
        label_parts = []
        if "name" in SHOW_FIELDS:
            label_parts.append(str(m))
        if "channels" in SHOW_FIELDS:
            label_parts.append(f"{sum(c1s)}→{c2}")
        if "n" in SHOW_FIELDS:
            label_parts.append(f"n={n_}")
        if "args" in SHOW_FIELDS and args:
            label_parts.append(str(args))
        if not label_parts:
            label_parts = [str(m)]

        # Mermaid 用 <br> 换行，并替换掉双引号
        label = "<br>".join(label_parts).replace('"', "'")

        layers.append(
            {
                "idx": i,
                "group": group,
                "from": from_idxs,
                "label": label,
                "type": str(m),
            }
        )
        ch.append(c2)

    # --- edges ---
    for node in layers:
        dst = node["idx"]
        for src in node["from"]:
            try:
                src_i = int(src)
            except:
                src_i = -1
            src_idx = src_i if src_i >= 0 else dst - 1
            edges.append((src_idx, dst))
    return layers, edges, len(seq_backbone)


# ---------------- Mermaid export ----------------
def draw_mermaid(
    layers: List[Dict], edges: List[Tuple[int, int]], backbone_len: int, out_file="model_graph"
):
    mermaid_lines: List[str] = []
    mermaid_lines.append("flowchart LR")
    mermaid_lines.append("    %% YAML -> Model 3-column graph")
    mermaid_lines.append('    Input(("Input<br>C=3")):::input')

    # --- Backbone / Neck / Head 分组 ---
    head_idx = layers[-1]["idx"] if layers else -1
    neck_layers = layers[backbone_len:-1] if backbone_len < len(layers) - 1 else []

    groups = {
        "Backbone": layers[:backbone_len],
        "Neck": neck_layers,
        "Head": [layers[-1]] if head_idx >= 0 else [],
    }

    group_node_ids: Dict[str, List[str]] = {}

    for name, nodes in groups.items():
        mermaid_lines.append(f"    subgraph {name}")
        mermaid_lines.append("        direction TB")
        ids: List[str] = []
        for node in nodes:
            nid = f"L{node['idx']}"
            ids.append(nid)
            mermaid_lines.append(
                f'        {nid}["{node["label"]}"]:::{name.lower()}'
            )
        mermaid_lines.append("    end")
        mermaid_lines.append("")
        group_node_ids[name] = ids

    # --- 不可见锚点，用于三列顶端对齐 ---
    mermaid_lines.append('    T1((" ")):::invis')
    mermaid_lines.append('    T2((" ")):::invis')
    mermaid_lines.append('    T3((" ")):::invis')
    mermaid_lines.append("")

    anchor_edges: List[Tuple[str, str]] = []
    if group_node_ids["Backbone"]:
        anchor_edges.append(("T1", group_node_ids["Backbone"][0]))
    if group_node_ids["Neck"]:
        anchor_edges.append(("T2", group_node_ids["Neck"][0]))
    if group_node_ids["Head"]:
        anchor_edges.append(("T3", group_node_ids["Head"][0]))

    # 先写锚点连线（后面用 linkStyle 隐藏）
    for src, dst in anchor_edges:
        mermaid_lines.append(f"    {src} --- {dst}")

    # --- 真实模型 Edges ---
    for src, dst in edges:
        src_id = f"L{src}" if src >= 0 else "Input"
        dst_id = f"L{dst}"
        mermaid_lines.append(f"    {src_id} --> {dst_id}")

    # --- 隐藏锚点连线（它们是最前面的几条边） ---
    if anchor_edges:
        idxs = ",".join(str(i) for i in range(len(anchor_edges)))
        mermaid_lines.append("")
        mermaid_lines.append(f"    linkStyle {idxs} stroke:none")

    # --- 样式定义 ---
    mermaid_lines.append("")
    mermaid_lines.append(
        "    classDef backbone fill:#C2E7D9,stroke:#2F4F4F,stroke-width:1px;"
    )
    mermaid_lines.append(
        "    classDef neck fill:#FFD6A5,stroke:#2F4F4F,stroke-width:1px;"
    )
    mermaid_lines.append(
        "    classDef head fill:#FFB5A7,stroke:#2F4F4F,stroke-width:1px;"
    )
    mermaid_lines.append(
        "    classDef input fill:#FFF0B3,stroke:#555,stroke-width:1px;"
    )
    mermaid_lines.append("    classDef invis fill:none,stroke:none;")

    # --- 保存 ---
    with open(f"{out_file}.mmd", "w", encoding="utf-8") as f:
        f.write("\n".join(mermaid_lines))
    print(f"✅ Mermaid graph saved to {out_file}.mmd")


# ---------------- main ----------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python yml2modelgraph_mermaid.py model.yaml [out_name]")
        return

    yml_path = sys.argv[1]
    out_name = sys.argv[2] if len(sys.argv) > 2 else "model_graph"

    with open(yml_path, "r") as f:
        d = yaml.safe_load(f)

    layers, edges, backbone_len = parse_yaml_model(d, input_channels=3)
    draw_mermaid(layers, edges, backbone_len, out_file=out_name)


if __name__ == "__main__":
    main()
