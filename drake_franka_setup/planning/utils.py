import numpy as np
import pydrake.all as pd
import pybezier as pb
from typing import List

def composite_bezier_to_drake(composite_curve : pb.CompositeBezierCurve):
    segments = []
    for c in composite_curve.curves:
        segments.append(pd.BezierCurve(c.initial_time,
                                    c.final_time,
                                    (c.points).astype(np.float32).T))

    return pd.CompositeTrajectory(segments)

def pybezier_to_drake_bezier_curve(c : pb.BezierCurve):
    return pd.BezierCurve(c.initial_time,
                       c.final_time,
                       (c.points).astype(np.float32).T)

def drake_composite_bezier_to_pybezier(traj : pd.CompositeTrajectory):
    num_segments = traj.get_number_of_segments()
    segs = []
    for seg_id in range(num_segments):
        seg = traj.segment(seg_id)
        assert isinstance(seg, pd.BezierCurve)
        segs.append(pb.BezierCurve(seg.control_points().T, seg.start_time(), seg.end_time()))
    return pb.CompositeBezierCurve(segs)

def add_stationary_start_and_end(trajectory: pd.CompositeTrajectory,
                                  start_duration: float = 0.1,
                                  end_duration: float = 0.1) -> pd.CompositeTrajectory:
    """
    Add stationary segments at both the start and end of a trajectory.

    This helps prevent lurching at trajectory boundaries by allowing the robot to settle
    at the initial position before starting motion, and smoothly come to rest at the end.

    Args:
        trajectory: CompositeTrajectory to modify
        start_duration: Duration of the starting stationary segment in seconds (default: 0.1s)
        end_duration: Duration of the ending stationary segment in seconds (default: 0.1s)

    Returns:
        CompositeTrajectory with stationary segments at both ends
    """
    # Get initial and final configurations
    first_segment = trajectory.segment(0)
    q_start = first_segment.value(first_segment.start_time())

    last_segment = trajectory.segment(trajectory.get_number_of_segments() - 1)
    q_end = last_segment.value(last_segment.end_time())

    # Create starting stationary segment
    start_segment = pd.BezierCurve(
        start_time=0.0,
        end_time=start_duration,
        control_points=np.column_stack([q_start, q_start])
    )

    # Time-shift all existing segments
    shifted_segments = [start_segment]
    for i in range(trajectory.get_number_of_segments()):
        seg = trajectory.segment(i)
        if isinstance(seg, pd.BezierCurve):
            shifted_segments.append(
                pd.BezierCurve(
                    start_time=seg.start_time() + start_duration,
                    end_time=seg.end_time() + start_duration,
                    control_points=seg.control_points()
                )
            )

    # Create ending stationary segment
    end_time = trajectory.end_time() + start_duration
    end_segment = pd.BezierCurve(
        start_time=end_time,
        end_time=end_time + end_duration,
        control_points=np.column_stack([q_end, q_end])
    )
    shifted_segments.append(end_segment)

    return pd.CompositeTrajectory(shifted_segments)


def waypoints_to_composite_bezier_smooth(waypoints, T_tot=10, tension=0.5):
    """
    Create a smooth composite Bezier curve using cardinal spline method.
    Better for closed curves and circular trajectories.
    
    Args:
        tension: 0 = tight corners, 1 = very smooth (default: 0.5)
    """
    waypoints = np.array(waypoints)
    num_waypoints, num_dims = waypoints.shape
    
    if num_waypoints < 2:
        raise ValueError("Need at least 2 waypoints")
    
    segments = []
    T_seg = T_tot / (num_waypoints - 1)
    
    for i in range(num_waypoints - 1):
        p0 = waypoints[i].copy()
        p3 = waypoints[i + 1].copy()
        
        # Get neighboring points for tangent calculation
        if i == 0:
            p_prev = waypoints[-1] if num_waypoints > 2 else p0  # Use last point for closed curve
        else:
            p_prev = waypoints[i - 1]
        
        if i == num_waypoints - 2:
            p_next = waypoints[0] if num_waypoints > 2 else p3  # Use first point for closed curve
        else:
            p_next = waypoints[i + 2]
        
        # Cardinal spline tangents
        tangent_start = (1 - tension) * (p3 - p_prev) / 2
        tangent_end = (1 - tension) * (p_next - p0) / 2
        
        # Control points
        p1 = p0 + tangent_start / 3
        p2 = p3 - tangent_end / 3
        
        control_points = np.column_stack([p0, p1, p2, p3])

        bezier_segment = pd.BezierCurve(
            start_time=i * T_seg,
            end_time=(i + 1) * T_seg,
            control_points=control_points
        )
        segments.append(bezier_segment)

    composite = pd.CompositeTrajectory(segments)
    return composite

import networkx as nx

def write_graph_summary(g: nx.Graph):
    summary = []
    summary.append(f"Graph Summary for Dynamic Roadmap")
    summary.append(f"=====================================")
    
    # Basic graph information
    summary.append(f"Number of nodes: {g.number_of_nodes()}")
    summary.append(f"Number of edges: {g.number_of_edges()}")
    
    # Connected components
    components = list(nx.connected_components(g))
    summary.append(f"Number of connected components: {len(components)}")
    print(f"component sizes {[len(c) for c in components]}")
    # Largest component details
    largest_component = max(components, key=len)
    summary.append(f"Largest component size: {len(largest_component)} nodes")
    
    # Degree information
    degrees = [d for n, d in g.degree()]
    avg_degree = np.mean(degrees)
    max_degree = max(degrees)
    min_degree = min(degrees)
    summary.append(f"Average node degree: {avg_degree:.2f}")
    summary.append(f"Maximum node degree: {max_degree}")
    summary.append(f"Minimum node degree: {min_degree}")
    # cut_vertices = list(nx.articulation_points(g))
    # summary.append(f"Number of articulation points (potential bottlenecks): {len(cut_vertices)}")
    # Isolated nodes
    isolated_nodes = list(nx.isolates(g))
    summary.append(f"Number of isolated nodes: {len(isolated_nodes)}")
    component_sizes = [len(c) for c in components]
    summary.append("Component size distribution:")
    summary.append(f"  Min: {min(component_sizes)}")
    summary.append(f"  Max: {max(component_sizes)}")
    summary.append(f"  Mean: {np.mean(component_sizes):.2f}")
    summary.append(f"  Median: {np.median(component_sizes):.2f}")
    print("\n".join(summary))

def pwl_path_length(path: List[np.ndarray]):
    length = 0
    for i in range(len(path)-1):
        length += np.linalg.norm(path[i]-path[i+1])
    return length

def arc_length_interpolate_path_preserve_knots(path: np.ndarray, 
                                               target_segment_length: float, 
                                               repeat_start_and_end: 0):
    """
    Alternative approach: specify target maximum segment length.
    All original knots are preserved, and segments longer than target_segment_length are subdivided.
    """
    assert path.shape[1] >= 2
    assert path.shape[0] >= 2
    
    segment_vectors = np.diff(path, axis=1)
    segment_lengths = np.linalg.norm(segment_vectors, axis=0)
    
    lerp = [path[:, 0]]  # Start with first point
    
    for i in range(len(segment_lengths)):
        segment_length = segment_lengths[i]
        
        if segment_length <= target_segment_length:
            # Segment is short enough, no subdivision needed
            pass
        else:
            # Subdivide this segment
            n_subsegments = int(np.ceil(segment_length / target_segment_length))
            t_values = np.linspace(0, 1, n_subsegments + 1)[1:-1]  # Exclude endpoints
            
            for t in t_values:
                interpolated_point = path[:, i] * (1 - t) + path[:, i + 1] * t
                lerp.append(interpolated_point)
        
        # Always add the next knot point
        lerp.append(path[:, i + 1])
    
    if repeat_start_and_end >0:
        for _ in range(repeat_start_and_end):
            lerp.insert(0, path[:,0])
            lerp.insert(-1, path[:,-1])
        
    return np.array(lerp).T