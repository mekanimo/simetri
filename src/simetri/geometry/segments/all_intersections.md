# How `all_ips_edges` Finds Intersections

## Purpose

`all_ips_edges` receives a collection of polygon-like shapes and returns every point where two of their edges intersect.

Each result has this form:

```text
((x, y), (edge_id_1, edge_id_2))
```

The edge IDs refer to the flattened list of all edges from all input shapes. The function does not return intersections between polygon interiors. It only tests pairs of edges.

## 1. Flatten the edges

The function first visits every shape and every edge in that shape. For each edge it extracts the first two coordinates of both endpoints. This supports both ordinary two-coordinate points and homogeneous points that contain an additional coordinate.

Every edge is stored as four numbers:

```text
[start_x, start_y, end_x, end_y]
```

The original position of the edge becomes its edge ID.

## 2. Compute bounding boxes

For each edge, the function calculates its axis-aligned bounding box:

```text
minimum x, minimum y, maximum x, maximum y
```

The box is the smallest upright rectangle that contains the edge.

Two edges can only intersect if their bounding boxes overlap. This is a cheap rejection test. If the boxes do not overlap, the more expensive line-intersection calculation is unnecessary.

## 3. Sort by the left side of the bounding box

The edges are sorted by their minimum x-coordinate. This creates a left-to-right processing order.

For the edge currently being processed, only edges later in the sorted array are considered. This ensures that each unordered pair is considered once instead of twice.

This is sweep-like, but it is not a full Bentley-Ottmann sweep line. The implementation does not maintain a balanced active-edge tree or schedule future intersection events. It scans the remaining suffix of the array and applies the bounding-box mask to that suffix.

## 4. Select possible candidates

For one current edge, NumPy compares its bounding box with the bounding boxes of all later edges. A later edge is retained when all four conditions are true:

- its minimum x is no greater than the current edge's maximum x;
- its maximum x is no less than the current edge's minimum x;
- its minimum y is no greater than the current edge's maximum y;
- its maximum y is no less than the current edge's minimum y.

The result is a NumPy array containing only possible intersecting edges.

The bounding-box test removes many pairs for inputs whose edges occupy separate regions. It does not guarantee a small candidate set. If many edges overlap in their bounding boxes, nearly every pair can remain a candidate.

## 5. Compute intersections in NumPy batches

For the current edge and all of its candidates, the function computes the segment direction vectors and the denominator of the two-line intersection formula in arrays.

A denominator close to zero means that the segments are parallel according to Simetri's configured relative and absolute tolerances. Those pairs are discarded at this stage.

For every nonparallel candidate, the algorithm computes two parameters:

- one parameter says where the intersection lies along the current edge;
- the other says where it lies along the candidate edge.

A pair intersects as a finite segment when both parameters are between zero and one, inclusive.

The intersection coordinates are then calculated from the first parameter and the current edge's direction vector. NumPy performs these calculations for the whole candidate batch at once.

## 6. Build the result list

For each valid pair, the function appends:

```text
((intersection_x, intersection_y), (original_edge_id_1, original_edge_id_2))
```

The sorted working array is only an internal representation. The saved edge IDs restore the IDs from the original flattened edge order.

The input shapes are not modified.

## What counts as an intersection?

The current calculation treats an intersection as valid when the two finite-segment parameters satisfy:

```text
0 <= parameter_on_first_edge <= 1
0 <= parameter_on_second_edge <= 1
```

Therefore, endpoint contacts are included. Parallel and collinear cases are excluded by the near-zero denominator test. The exact boundary behavior near parallelism is controlled by Simetri's configured tolerances.

## Complexity

Let:

- `E` be the total number of edges;
- `C` be the number of pairs that survive the bounding-box test;
- `K` be the number of reported intersections.

The main costs are:

1. extracting the edges and computing their boxes: `O(E)`;
2. sorting by minimum x: `O(E log E)`;
3. scanning the suffix for every edge: `O(E^2)` in the current implementation;
4. calculating intersections for surviving candidates: `O(C)`;
5. constructing the output: `O(K)`.

Consequently, the worst-case complexity is:

```text
O(E^2 + C + K)
```

Since `C` can itself be `O(E^2)`, this is normally summarized as `O(E^2)` worst case.

The NumPy batching improves the constant factor of the arithmetic and reduces Python-level function-call overhead. It does not change the worst-case complexity, because the suffix scans are still quadratic.

## Difference from a true sweep-line algorithm

A true output-sensitive Bentley-Ottmann-style algorithm maintains:

- an event queue ordered by x-coordinate;
- an active set ordered by the edges' vertical positions at the sweep line;
- neighbor-intersection events that are inserted into the event queue.

With suitable assumptions and data structures, that approach targets `O(E log E + C + K)`. The current `all_ips_edges` function does not maintain those structures. Its left-to-right sorting and bounding-box filtering are useful optimizations, but calling it a full sweep-line algorithm would be inaccurate.
