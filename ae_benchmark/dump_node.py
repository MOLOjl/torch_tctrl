import igraph as ig
import sys
import os
import re
import math
from typing import List, Tuple
from pathlib import Path
import ast

def extract_log_filenames(csv_path, max_logs=4) -> List[Tuple[int, str]]:
    """
    从 csv 文件中提取 (device, log_file_name)
    最多提取 max_logs 个，达到后提前结束读取
    返回: List[(rank, log_filename)]
    """
    results = []

    pattern = re.compile(
        r'^device\s*,\s*(\d+)\s*,\s*log_file_name\s*,\s*[“"]([^”"]+)[”"]'
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(results) >= max_logs:
                break

            line = line.strip()
            m = pattern.match(line)
            if m:
                rank = int(m.group(1))
                log_file = m.group(2)
                results.append((rank, log_file))

    return results

def parse_centrality_nodes(graph, rank, percent: int = 5, out_dir="../logs"):
    """
    计算 degree / closeness / betweenness 中心性，
    并输出前 percent% 的节点 id
    """
    n = graph.vcount()
    if n == 0:
        print("计算图中没有节点，无法生成百分比文件")
        return

    # ----------- 中心性计算 -----------
    if n > 1:
        degree_centrality = [d / (n - 1) for d in graph.degree()]
    else:
        degree_centrality = [0.0]

    centrality_list = [
        ("dc", degree_centrality),
        ("cc", graph.closeness()),
        # ("bc", graph.betweenness()),
        ("bc", graph.betweenness(cutoff=5)),
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------- 排序 & 输出 -----------
    for tag, centrality in centrality_list:
        total_nodes = len(centrality)
        if total_nodes == 0:
            print(f"{tag}: 没有节点，跳过")
            continue

        # 按中心性值降序排序的 vertex id
        sorted_vertex_ids: List[int] = sorted(
            range(total_nodes),
            key=lambda v: centrality[v],
            reverse=True,
        )

        # 计算前 percent% 的节点数（向上取整）
        cur_count = math.ceil(percent / 100.0 * total_nodes)
        node_ids = sorted_vertex_ids[:cur_count]

        out_path = out_dir / f"{tag}_p{percent}_nodes_rank{rank}.txt"
        with open(out_path, "w") as f:
            for nid in node_ids:
                f.write(f"{nid}\n")

        print(
            f"基于 {tag}，已保存前 {percent}% 的节点 "
            f"（共 {len(node_ids)}/{n} 个） → {out_path}"
        )

def parse_compute_graph1(input_log):
    nodes = set()
    edges = []

    with open(input_log, 'r') as f:
        for line in f:
            row = ast.literal_eval(line)

            if row.get('INSTRUCTION') != 'INSTRUCTION':
                continue
            if row['name'] == 'add_' and row['inputs'][0] == row['outputs'][0]:
                continue

            inputs  = [int(i[1:]) for i in row['inputs']]
            outputs = [int(o[1:]) for o in row['outputs']]

            for src in inputs:
                nodes.add(src)
                for dst in outputs:
                    nodes.add(dst)
                    edges.append((
                        src, dst,
                        row['name'],
                        row['compute_cost'],
                        row['mem_cost'],
                    ))

    max_node = max(nodes) if nodes else 0

    g = ig.Graph(directed=True)
    g.add_vertices(max_node + 1)
    g.add_edges([(e[0], e[1]) for e in edges])

    g.es['op']   = [e[2] for e in edges]
    g.es['cost'] = [e[3] for e in edges]
    g.es['mem']  = [e[4] for e in edges]

    return g

def parse_compute_graph(input_log):
    r"""
    读取计算过程中的op日志, 解析出对应的计算图
    把节点按照介数中心性从高到低排序，并按 5% 的比例保存到e2_c5.txt中
    """
    # ------------------- 原有解析部分（略作整理） -------------------
    with open(input_log, 'r') as f:
        data = f.readlines()
        data = [eval(row.replace('\n', '')) for row in data]

    nodes = set()
    edges = []
    nodes_times = {}
    
    for row in data:
        if row.get('INSTRUCTION') != 'INSTRUCTION':
            continue
        if row['name'] == 'add_' and row['inputs'][0] == row['outputs'][0]:
            continue

        for iid in row['inputs']:
            input_id = int(iid[1:])
            nodes.add(input_id)
            nodes_times.setdefault(input_id, 0)

            for oid in row['outputs']:
                output_id = int(oid[1:])
                nodes_times[input_id] += 1
                edges.append((input_id, output_id,
                              {'op': row['name'],
                               'cost': row['compute_cost'],
                               'mem': row['mem_cost']}))

        for oid in row['outputs']:
            output_id = int(oid[1:])
            nodes.add(output_id)
            nodes_times.setdefault(output_id, 0)
            for iid in row['inputs']:
                nodes_times[output_id] += 1

    # ------------------- 构建 igraph 图 -------------------
    g = ig.Graph()
    g.add_vertices(max(list(nodes)) + 1)
    for src, dst, attr in edges:
        g.add_edge(src, dst, **attr)

    return g

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <csv> or python {sys.argv[0]} <log_path> <rank>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    # -------- 情况 1：CSV 文件 --------
    if input_path.endswith(".csv"):
        results = extract_log_filenames(input_path)

        print(f"Input CSV: {input_path}")
        print(f"Extracted results: {results}")

        for rank, log_file in results:
            g = parse_compute_graph1(log_file)
            parse_centrality_nodes(g, rank)
    
    # -------- 情况 2：单个 .log 文件 --------
    elif input_path.endswith(".log"):
        if(len(sys.argv) < 3):
            print(f"Usage: python {sys.argv[0]} <log_path> <rank>")
            sys.exit(1)
        rank = sys.argv[2]
        print(f"Input log: {input_path}, rank {rank}")
        g = parse_compute_graph1(input_path)
        parse_centrality_nodes(g, rank)
    else:
        raise ValueError(
            f"Unsupported input format: {input_path} "
            "(expect .csv or .log)"
        )

if __name__ == '__main__':
    main()