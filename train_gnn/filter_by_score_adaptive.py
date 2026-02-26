import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# Hyperparameter: minimum connections per node
MIN_CONNECTIONS = 20
PATH_LOAD = 'path/to/9606.protein.links.ensembl.txt'
PATH_SAVE = 'path/to/9606.protein.links.ensembl_{thresh}_keep{min_connections}_adaptive.txt'

# Load graph
print("Loading graph...")
graph_df = pd.read_csv(PATH_LOAD, sep='\t')
print(f"Total edges in input: {len(graph_df)}")

# Pre-compute all nodes
all_nodes = set(graph_df['protein1'].unique()) | set(graph_df['protein2'].unique())
print(f"Total nodes: {len(all_nodes)}")

# Count original connections for each node using vectorized operations
print("Computing original connection counts...")
original_connections = Counter(graph_df['protein1']) + Counter(graph_df['protein2'])

# Create edge lookup dictionary once using itertuples (much faster)
print("Building edge lookup index...")
edge_data = {}
for row in graph_df.itertuples(index=False):
    edge_data[(row.protein1, row.protein2)] = {
        'protein1': row.protein1,
        'protein2': row.protein2,
        'combined_score': row.combined_score
    }

# Filter by score thresholds
thresholds = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

for threshold in tqdm(thresholds, desc="Processing thresholds"):
    print(f"\n=== Processing threshold {threshold} ===")

    # Split edges by threshold using boolean indexing (much faster than copy)
    above_mask = graph_df['combined_score'] >= threshold
    above_threshold = graph_df[above_mask]
    below_threshold = graph_df[~above_mask]

    print(f"Edges above threshold: {len(above_threshold)}")
    print(f"Edges below threshold: {len(below_threshold)}")

    # Count connections above threshold using vectorized operations
    connections_above = Counter(above_threshold['protein1']) + Counter(above_threshold['protein2'])

    # Build index of below-threshold edges by node (much faster lookup)
    below_by_node = defaultdict(list)
    below_edge_to_idx = {}  # Map (p1, p2) to index for fast reverse lookup
    for idx, row in zip(below_threshold.index, below_threshold.itertuples(index=False)):
        p1, p2, score = row.protein1, row.protein2, row.combined_score
        below_by_node[p1].append((p2, score, idx))
        below_by_node[p2].append((p1, score, idx))
        below_edge_to_idx[(p1, p2)] = idx

    # Sort each node's edges by score descending
    for node in below_by_node:
        below_by_node[node].sort(key=lambda x: x[1], reverse=True)

    # Track which edges to rescue
    rescued_indices = set()

    # For each node, rescue edges if needed
    for node in tqdm(all_nodes, desc=f"Checking nodes (threshold={threshold})", leave=False):
        current_count = connections_above.get(node, 0)
        original_count = original_connections[node]
        target_count = min(MIN_CONNECTIONS, original_count)

        if current_count < target_count:
            needed = target_count - current_count
            node_edges = below_by_node.get(node, [])

            # Take top 'needed' edges
            for neighbor, score, idx in node_edges[:needed]:
                rescued_indices.add(idx)

                # Also rescue the reverse edge if it exists in below_threshold
                reverse_key = (neighbor, node)
                if reverse_key in below_edge_to_idx:
                    rescued_indices.add(below_edge_to_idx[reverse_key])

    # Combine above-threshold edges with rescued edges
    if rescued_indices:
        rescued_df = below_threshold.loc[list(rescued_indices)]
        filtered = pd.concat([above_threshold, rescued_df], ignore_index=True)
        filtered = filtered.drop_duplicates(subset=['protein1', 'protein2'])
        print(f"Rescued {len(rescued_df)} edges to maintain minimum connections")
    else:
        filtered = above_threshold.copy()
        print("No edges needed to be rescued")

    # Ensure undirected (bidirectional)
    # Create set of edges efficiently
    filtered_edges = set(zip(filtered['protein1'], filtered['protein2']))
    missing_reverse = {(p1, p2) for (p1, p2) in filtered_edges if (p2, p1) not in filtered_edges}

    if len(missing_reverse) > 0:
        print(f"Making graph undirected: adding {len(missing_reverse)} reverse edges")

        # Create reverse edges
        reverse_rows = []
        for p1, p2 in tqdm(missing_reverse, desc="Adding reverse edges", leave=False):
            # Try to find in original graph
            if (p2, p1) in edge_data:
                reverse_rows.append(edge_data[(p2, p1)])
            else:
                # Create by swapping (should be rare)
                orig = edge_data.get((p1, p2))
                if orig:
                    reverse_rows.append({
                        'protein1': p2,
                        'protein2': p1,
                        'combined_score': orig['combined_score']
                    })

        if reverse_rows:
            reverse_df = pd.DataFrame(reverse_rows)
            filtered = pd.concat([filtered, reverse_df], ignore_index=True)
            filtered = filtered.drop_duplicates(subset=['protein1', 'protein2'])

    # Final bidirectionality verification
    final_edges = set(zip(filtered['protein1'], filtered['protein2']))
    still_missing = {(p1, p2) for (p1, p2) in final_edges if (p2, p1) not in final_edges}

    if len(still_missing) > 0:
        print(f"WARNING: Graph still has {len(still_missing)} unidirectional edges!")
    else:
        print("✓ Graph is fully bidirectional")

    # Final connection count verification using vectorized operations
    final_connections = Counter(filtered['protein1']) + Counter(filtered['protein2'])
    nodes_below_min = sum(1 for node in all_nodes if final_connections.get(node, 0) < min(MIN_CONNECTIONS, original_connections[node]))
    print(f"Nodes below target minimum: {nodes_below_min}/{len(all_nodes)}")

    output_path = PATH_SAVE.format(thresh=threshold, min_connections=MIN_CONNECTIONS)
    filtered.to_csv(output_path, sep='\t', index=False)

    total_edges = len(filtered)
    print(f"Total edges retained: {total_edges} ({total_edges/len(graph_df)*100:.2f}%)")
    print(f"Saved to: {output_path}")
