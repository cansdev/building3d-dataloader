#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2025-09-15
# @Author  : Batuhan Arda Bekar
# @File    : weight_assignment.py
# @Purpose : Point weight assignment based on geometric features (edges and vertices)

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_border_weights(points, k_neighbors=30, normalize_weights=True, multi_scale=False):
    """
    Compute border weights for all points based on local geometric analysis.
    Highlights object borders (edges and corners) in a unified way.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        k_neighbors (int): Number of neighbors for local analysis
        normalize_weights (bool): Whether to normalize weights to [0, 1]
        multi_scale (bool): Whether to use multi-scale analysis
    Returns:
        np.ndarray: border_weights of shape (N,)
    """
    n_points = points.shape[0]
    if n_points < 4:
        return np.zeros(n_points)
    k_neighbors = min(k_neighbors, n_points - 1)
    if multi_scale:
        scales = [max(10, k_neighbors // 2), k_neighbors, min(n_points - 1, k_neighbors * 2)]
        border_weights_all = []
        for scale in scales:
            if scale >= n_points:
                continue
            eigenvalues = calculate_local_features(points, scale)
            edge_w = calculate_edge_weights(eigenvalues)
            vertex_w = calculate_vertex_weights(eigenvalues)
            border_w = np.maximum(edge_w, vertex_w)
            border_weights_all.append(border_w)
        if border_weights_all:
            scale_weights = [0.2, 0.5, 0.3][:len(border_weights_all)]
            border_weights = np.average(border_weights_all, axis=0, weights=scale_weights)
        else:
            border_weights = np.zeros(n_points)
    else:
        eigenvalues = calculate_local_features(points, k_neighbors)
        edge_w = calculate_edge_weights(eigenvalues)
        vertex_w = calculate_vertex_weights(eigenvalues)
        border_weights = np.maximum(edge_w, vertex_w)
    border_weights = enhance_weights(border_weights, weight_type="border")
    border_weights = filter_isolated_features(points, border_weights, min_neighbors=3, radius=0.08)
    if normalize_weights:
        border_weights = normalize_weight_values(border_weights)
    return border_weights


def calculate_local_features(points, k_neighbors):
    """
    Calculate eigenvalues of local covariance matrices for all points.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        k_neighbors (int): Number of neighbors for local analysis
        
    Returns:
        np.ndarray: Eigenvalues with shape (N, 3) sorted in descending order
    """
    n_points = points.shape[0]
    eigenvalues = np.zeros((n_points, 3))
    
    # Build neighbor tree
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='kd_tree').fit(points)
    
    for i in range(n_points):
        # Get k-nearest neighbors (including self)
        distances, indices = nbrs.kneighbors([points[i]])
        neighbor_points = points[indices[0]]  # Shape: (k+1, 3)
        
        # Center the neighborhood
        centroid = np.mean(neighbor_points, axis=0)
        centered_points = neighbor_points - centroid
        
        # Compute covariance matrix
        if len(centered_points) > 1:
            cov_matrix = np.cov(centered_points.T)
            
            # Handle edge case where covariance is not positive definite
            try:
                eigvals = np.linalg.eigvals(cov_matrix)
                # Sort eigenvalues in descending order
                eigvals = np.sort(np.real(eigvals))[::-1]
                
                # Ensure non-negative eigenvalues
                eigvals = np.maximum(eigvals, 0)
                eigenvalues[i] = eigvals
                
            except np.linalg.LinAlgError:
                # Fallback for degenerate cases
                eigenvalues[i] = [0, 0, 0]
        else:
            eigenvalues[i] = [0, 0, 0]
    
    return eigenvalues


def calculate_edge_weights(eigenvalues):
    """
    Enhanced edge weight calculation for building structures, optimized for sloped/cross lines.
    Detects linear features including diagonal edges, roof ridges, and structural intersections.
    
    Args:
        eigenvalues (np.ndarray): Eigenvalues with shape (N, 3)
        
    Returns:
        np.ndarray: Edge weights with shape (N,)
    """
    lambda1, lambda2, lambda3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    
    # Avoid division by zero
    epsilon = 1e-8
    lambda1_safe = lambda1 + epsilon
    lambda2_safe = lambda2 + epsilon
    lambda3_safe = lambda3 + epsilon
    
    # Building edges: λ1 >> λ2 > λ3 (linear with some thickness)
    
    # Measure 1: Enhanced linearity for sloped edges
    # Traditional linearity measure
    linearity_basic = (lambda1 - lambda2) / lambda1_safe
    linearity_basic = np.clip(linearity_basic, 0, 1)
    
    # Enhanced linearity for sloped/diagonal edges
    # Check for strong dominant direction regardless of orientation
    eigenvalue_dominance = lambda1_safe / (lambda1_safe + lambda2_safe + lambda3_safe)
    linearity_enhanced = linearity_basic * eigenvalue_dominance
    
    # Adaptive linearity threshold for sloped lines
    # Sloped edges may have slightly less clear linearity due to point sampling
    linearity_adaptive = np.where(eigenvalue_dominance > 0.6, 
                                  linearity_basic * 1.2,  # Boost strong linear features
                                  linearity_basic * 0.9)  # Slight penalty for weak features
    linearity = np.maximum(linearity_enhanced, linearity_adaptive)
    linearity = np.clip(linearity, 0, 1)
    
    # Measure 2: Improved thickness handling for cross/sloped lines
    thickness_ratio = lambda2_safe / lambda1_safe
    
    # Cross lines and junctions may have higher λ2 due to intersection geometry
    # Adaptive thickness scoring based on eigenvalue pattern
    cross_line_indicator = (lambda2_safe > 0.3 * lambda1_safe) & (lambda3_safe < 0.2 * lambda2_safe)
    
    # Enhanced edge cross detection - more specific pattern for intersections
    edge_cross_pattern = (lambda2_safe > 0.25 * lambda1_safe) & (lambda2_safe < 0.7 * lambda1_safe) & (lambda3_safe < 0.15 * lambda2_safe)
    
    # Standard thickness scoring for regular edges
    thickness_score_std = np.where(thickness_ratio < 0.4, thickness_ratio * 2.5, 1.0 - thickness_ratio)
    
    # Enhanced scoring for potential cross/junction points
    thickness_score_cross = np.where(thickness_ratio < 0.6, 
                                    thickness_ratio * 1.8 + 0.2,  # More tolerant
                                    1.2 - thickness_ratio)        # Gentler penalty
    
    # Special scoring for edge crosses - even more tolerant
    thickness_score_edge_cross = np.where(thickness_ratio < 0.65,
                                         thickness_ratio * 1.5 + 0.3,  # Very tolerant for edge crosses
                                         1.3 - thickness_ratio)        # Very gentle penalty
    
    # Choose appropriate thickness scoring
    thickness_score = np.where(edge_cross_pattern, thickness_score_edge_cross,
                              np.where(cross_line_indicator, thickness_score_cross, thickness_score_std))
    thickness_score = np.clip(thickness_score, 0, 1)
    
    # Measure 3: Adaptive third dimension handling
    third_dim_ratio = lambda3_safe / lambda1_safe
    
    # For sloped edges in 3D, λ3 might be slightly larger due to spatial orientation
    # Use adaptive suppression based on the overall eigenvalue pattern
    slope_tolerance = np.where(eigenvalue_dominance > 0.65, 
                              0.15,  # More tolerant for strong linear features
                              0.10)  # Standard tolerance
    
    third_dim_suppression = np.where(third_dim_ratio < slope_tolerance,
                                    1.0 - third_dim_ratio * 5,    # Gentle penalty
                                    1.0 - third_dim_ratio * 8)    # Stronger penalty
    third_dim_suppression = np.clip(third_dim_suppression, 0, 1)
    
    # Measure 4: Enhanced cross-line and junction detection
    # Detect potential line intersections and complex edge geometry
    # Edge crosses have characteristic eigenvalue patterns: λ1 dominant, λ2 moderate, λ3 small
    edge_cross_indicator = (lambda2_safe > 0.25 * lambda1_safe) & (lambda2_safe < 0.7 * lambda1_safe) & (lambda3_safe < 0.15 * lambda2_safe)
    
    # Enhanced junction scoring with stronger emphasis on edge crosses
    junction_score = np.where(edge_cross_indicator & (eigenvalue_dominance > 0.5),
                             0.9 + 0.3 * eigenvalue_dominance,  # Strong boost for edge crosses
                             np.where(cross_line_indicator & (eigenvalue_dominance > 0.5),
                                     0.8 + 0.2 * eigenvalue_dominance,  # Boost junction areas
                                     0.5))  # Neutral for regular edges
    
    # Measure 5: Slope-aware edge quality
    # Consider the 3D orientation and slope characteristics
    slope_awareness = np.minimum(1.0, eigenvalue_dominance * 1.5)
    
    # More sophisticated combination emphasizing edge crosses
    edge_weights = (0.40 * linearity +           # Slightly reduced core linearity weight
                   0.25 * thickness_score +      # Reduced thickness importance  
                   0.12 * third_dim_suppression +# Keep 3D suppression
                   0.15 * junction_score +       # Increased junction boost for edge crosses
                   0.08 * slope_awareness)       # Increased slope awareness
    
    # Stricter filtering to reduce false positives, but more tolerant for edge crosses
    # Require stronger linearity for edge detection
    min_linearity_threshold = np.where(edge_cross_pattern, 0.3,  # Very tolerant for edge crosses
                                      np.where(eigenvalue_dominance > 0.7, 0.4, 0.6))  # Higher thresholds for others
    min_linearity_mask = linearity < min_linearity_threshold
    edge_weights[min_linearity_mask] *= np.where(edge_cross_pattern[min_linearity_mask], 0.6, 0.2)  # Less penalty for edge crosses
    
    # Stricter thickness filtering, but tolerant for edge crosses
    thick_threshold = np.where(edge_cross_pattern, 0.75,  # Very tolerant for edge crosses
                              np.where(cross_line_indicator, 0.65, 0.55))  # Stricter thresholds for others
    thick_mask = thickness_ratio > thick_threshold
    edge_weights[thick_mask] *= np.where(edge_cross_pattern[thick_mask], 0.7, 0.3)  # Less penalty for edge crosses
    
    # Enhanced isotropic suppression for false positive reduction
    isotropic_mask = (thickness_ratio > 0.7) & (third_dim_ratio > 0.4) & (eigenvalue_dominance < 0.5)
    edge_weights[isotropic_mask] *= 0.1  # Stronger suppression
    
    # Only boost very clear edges to reduce false positives
    clear_edge_mask = (linearity > 0.75) & (eigenvalue_dominance > 0.75) & (thickness_ratio < 0.4)
    edge_weights[clear_edge_mask] *= 1.15  # Reduced boost
    
    # Enhanced cross/junction detection with stronger emphasis on edge crosses
    enhanced_cross_junction_mask = edge_cross_indicator & (eigenvalue_dominance > 0.65) & (linearity > 0.5)
    edge_weights[enhanced_cross_junction_mask] *= 1.3  # Strong boost for edge crosses
    
    # Standard cross/junction detection  
    cross_junction_mask = cross_line_indicator & (eigenvalue_dominance > 0.75) & (linearity > 0.6)
    edge_weights[cross_junction_mask] *= 1.1  # Moderate boost for general junctions
    
    # Conservative slope enhancement - only for very clear cases
    slope_edge_mask = (eigenvalue_dominance > 0.8) & (third_dim_ratio < 0.15) & (linearity > 0.7)
    edge_weights[slope_edge_mask] *= 1.05  # Minimal boost
    
    # Add additional false positive filtering
    # Suppress areas with too uniform eigenvalues (noise/surfaces)
    uniform_eigenvalues = (lambda2_safe > 0.8 * lambda1_safe) & (lambda3_safe > 0.8 * lambda2_safe)
    edge_weights[uniform_eigenvalues] *= 0.05
    
    # Suppress very small eigenvalues (noise)
    small_eigenvalues = lambda1_safe < 0.01 * np.max(lambda1_safe)
    edge_weights[small_eigenvalues] = 0
    
    # Conservative final enhancement
    edge_weights = np.power(edge_weights, 1.3)  # Slightly more aggressive to suppress low values
    
    return np.clip(edge_weights, 0, 1)


def calculate_vertex_weights(eigenvalues):
    """
    Improved Harris Corner Detection for 3D point clouds with adaptive thresholding.
    Addresses missing corners through better parameter selection and multi-criteria analysis.
    
    Args:
        eigenvalues (np.ndarray): Eigenvalues with shape (N, 3)
        
    Returns:
        np.ndarray: Vertex weights with shape (N,)
    """
    lambda1, lambda2, lambda3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    
    # Avoid division by zero
    epsilon = 1e-8
    lambda1_safe = lambda1 + epsilon
    lambda2_safe = lambda2 + epsilon
    lambda3_safe = lambda3 + epsilon
    
    # Enhanced Harris Corner Response with multiple variants
    
    # Variant 1: Classic Harris Response
    determinant = lambda1_safe * lambda2_safe * lambda3_safe
    trace = lambda1_safe + lambda2_safe + lambda3_safe
    k = 0.04
    harris_classic = determinant - k * (trace ** 2)
    
    # Variant 2: Modified Harris for 3D corners (focus on two main directions)
    # For building corners: two significant eigenvalues matter most
    determinant_2d = lambda1_safe * lambda2_safe  # Focus on top 2 eigenvalues
    trace_2d = lambda1_safe + lambda2_safe
    harris_2d = determinant_2d - k * (trace_2d ** 2)
    
    # Variant 3: Shi-Tomasi response (minimum eigenvalue approach)
    # Use the second smallest eigenvalue as corner strength
    shi_tomasi = lambda2_safe
    
    # Variant 4: Noble's harmonic mean approach
    noble = (2 * determinant) / (trace + epsilon)
    
    # Variant 5: Förstner corner strength (balanced eigenvalues)
    # Good corners have balanced but distinct eigenvalues
    forstner = (lambda1_safe * lambda2_safe) / ((lambda1_safe + lambda2_safe) ** 2 + epsilon)
    
    # Variant 6: Building-specific corner measure
    # Check for L-shaped or T-shaped corner patterns
    ratio_21 = lambda2_safe / lambda1_safe
    ratio_32 = lambda3_safe / lambda2_safe
    
    # Building corners often have: moderate λ2/λ1 ratio, smaller λ3/λ2 ratio
    # Enhanced building corner detection - more permissive for 3D corners
    building_corner = ratio_21 * np.maximum(0.3, (1.0 - ratio_32))  # Allow some λ3 contribution
    
    # Combine all measures with adaptive weights
    combined_response = (0.2 * harris_classic +     # Classic Harris
                        0.25 * harris_2d +          # 2D-focused Harris  
                        0.2 * shi_tomasi +          # Minimum eigenvalue
                        0.1 * noble +               # Harmonic mean
                        0.15 * forstner +           # Balanced eigenvalues
                        0.1 * building_corner)      # Building-specific
    
    # Normalize each component separately to prevent one from dominating
    vertex_weights = np.zeros_like(combined_response)
    
    # Process each component with its own normalization
    components = [harris_classic, harris_2d, shi_tomasi, noble, forstner, building_corner]
    weights = [0.2, 0.25, 0.2, 0.1, 0.15, 0.1]
    
    for component, weight in zip(components, weights):
        if np.max(component) > 0:
            normalized_component = component / np.max(component)
            vertex_weights += weight * normalized_component
    
    # Adaptive thresholding based on data distribution - MUCH MORE PERMISSIVE
    if np.max(vertex_weights) > 0:
        # Use much lower percentile-based thresholding to keep more corners
        threshold = np.percentile(vertex_weights[vertex_weights > 0], 85)  # Top 15% instead of top 40%
        # Also use a minimum absolute threshold to catch medium-strength corners
        abs_threshold = 0.15 * np.max(vertex_weights)  # Accept corners with 15% of max response
        combined_threshold = min(threshold, abs_threshold)
        vertex_weights = np.where(vertex_weights > combined_threshold, vertex_weights, 0)
    
    # Additional enhancement for missed corners
    # Look for points with good corner characteristics even if response is moderate
    
    # Criterion 1: Two significant eigenvalues (corner intersection)
    two_significant = (lambda2_safe > 0.2 * lambda1_safe) & (lambda1_safe > 0.1 * np.max(lambda1))
    
    # Criterion 2: Reasonable 3D structure
    good_3d_structure = (lambda3_safe > 0.05 * lambda1_safe) & (lambda3_safe < 0.4 * lambda2_safe)
    
    # Criterion 3: Not too linear or too planar
    not_too_linear = lambda2_safe > 0.15 * lambda1_safe
    not_too_planar = lambda3_safe > 0.08 * lambda2_safe
    
    # Boost points that meet corner criteria but might have lower response
    corner_candidates = two_significant & good_3d_structure & not_too_linear & not_too_planar
    vertex_weights[corner_candidates] = np.maximum(vertex_weights[corner_candidates], 0.3)
    
    # Remove clearly inappropriate points - MUCH MORE PERMISSIVE
    # Too linear: very small λ2 relative to λ1 - relaxed threshold
    too_linear = lambda2_safe < 0.03 * lambda1_safe  # Relaxed from 0.05 to 0.03
    vertex_weights[too_linear] = 0
    
    # Too isotropic: all eigenvalues very similar - relaxed
    too_isotropic = (ratio_21 > 0.95) & (ratio_32 > 0.95)  # Relaxed from 0.9 to 0.95
    vertex_weights[too_isotropic] = 0
    
    # Too planar: very small λ3 - MUCH MORE PERMISSIVE for building corners
    # Building corners are often primarily 2D, so allow very small λ3
    too_planar = lambda3_safe < 0.005 * lambda2_safe  # Much more permissive: 0.5% instead of 2%
    vertex_weights[too_planar] = 0
    
    # Final enhancement with gentle power law
    vertex_weights = np.where(vertex_weights > 0, 
                             np.power(vertex_weights, 0.8),  # Gentle enhancement
                             0)
    
    # Ensure reasonable distribution
    if np.max(vertex_weights) > 0:
        vertex_weights = vertex_weights / np.max(vertex_weights)
    
    return vertex_weights


def enhance_weights(weights, weight_type="border", enhancement_factor=2.0):
    """
    Apply non-linear transformations to enhance border weight discrimination.
    Args:
        weights (np.ndarray): Raw weights
        weight_type (str): Only "border" supported
        enhancement_factor (float): Factor for enhancement
    Returns:
        np.ndarray: Enhanced weights
    """
    # More conservative enhancement to reduce false positives
    # Use higher power to suppress low weights more aggressively
    enhanced = np.power(weights, 1.2)  # More aggressive suppression of weak signals
    return enhanced


def filter_isolated_features(points, weights, min_neighbors=3, radius=0.1):
    """
    Filter out isolated high-weight points that don't have supporting neighbors.
    
    Args:
        points (np.ndarray): Point coordinates
        weights (np.ndarray): Point weights
        min_neighbors (int): Minimum number of high-weight neighbors required
        radius (float): Search radius for neighbors
        
    Returns:
        np.ndarray: Filtered weights
    """
    if len(points) < min_neighbors:
        return weights
        
    filtered_weights = weights.copy()
    
    # Build neighbor tree
    nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(points)
    
    # More conservative threshold for false positive reduction
    weight_threshold = 0.5  # Lowered from 0.6 to catch more potential false positives
    
    for i, weight in enumerate(weights):
        if weight > weight_threshold:
            # Find neighbors within radius
            neighbor_indices = nbrs.radius_neighbors([points[i]], return_distance=False)[0]
            neighbor_indices = neighbor_indices[neighbor_indices != i]  # Exclude self
            
            if len(neighbor_indices) > 0:
                # Count high-weight neighbors
                neighbor_weights = weights[neighbor_indices]
                high_weight_neighbors = np.sum(neighbor_weights > weight_threshold)
                
                # If not enough high-weight neighbors, reduce weight more aggressively
                if high_weight_neighbors < min_neighbors:
                    filtered_weights[i] *= 0.2  # More aggressive reduction
            else:
                # No neighbors at all, likely isolated noise
                filtered_weights[i] *= 0.1  # More aggressive reduction
    
    return filtered_weights


def normalize_weight_values(weights):
    """
    Normalize weights to [0, 1] range using robust statistics.
    
    Args:
        weights (np.ndarray): Raw weights
        
    Returns:
        np.ndarray: Normalized weights
    """
    if len(weights) == 0:
        return weights
    
    # Use percentile-based normalization for robustness
    p5, p95 = np.percentile(weights, [5, 95])
    
    if p95 - p5 > 1e-8:
        normalized = (weights - p5) / (p95 - p5)
        # Clamp to [0, 1]
        normalized = np.clip(normalized, 0, 1)
    else:
        # All weights are similar, return uniform
        normalized = np.ones_like(weights) * 0.5
    
    return normalized



def print_border_weight_statistics(border_weights):
    """
    Print detailed statistics about the computed border weights.
    Args:
        border_weights (np.ndarray): Border weights
    """
    print(f"\n=== Border Weight Statistics ===")
    border_mean = np.mean(border_weights)
    border_std = np.std(border_weights)
    border_min, border_max = np.min(border_weights), np.max(border_weights)
    border_high = np.sum(border_weights > 0.7)
    border_medium = np.sum((border_weights > 0.3) & (border_weights <= 0.7))
    border_low = np.sum(border_weights <= 0.3)
    print(f"   Range: [{border_min:.4f}, {border_max:.4f}]")
    print(f"   Mean±Std: {border_mean:.4f}±{border_std:.4f}")
    print(f"   Distribution: High({border_high}), Medium({border_medium}), Low({border_low})")


def visualize_border_weight_distribution(border_weights, save_path=None):
    """
    Create histogram plot of border weight distribution for analysis.
    Args:
        border_weights (np.ndarray): Border weights
        save_path (str): Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        plt.hist(border_weights, bins=50, alpha=0.8, color='orange', edgecolor='black')
        plt.title('Border Weight Distribution')
        plt.xlabel('Border Weight')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("Matplotlib not available for border weight distribution visualization")


def get_high_weight_points(points, edge_weights, vertex_weights, 
                          edge_threshold=0.7, vertex_threshold=0.7):
    """
    Extract points with high edge or vertex weights for analysis.
    
    Args:
        points (np.ndarray): Point coordinates
        edge_weights (np.ndarray): Edge weights
        vertex_weights (np.ndarray): Vertex weights
        edge_threshold (float): Threshold for high edge weights
        vertex_threshold (float): Threshold for high vertex weights
        
    Returns:
        dict: Dictionary containing high-weight point information
    """
    high_edge_mask = edge_weights > edge_threshold
    high_vertex_mask = vertex_weights > vertex_threshold
    
    result = {
        'high_edge_points': points[high_edge_mask],
        'high_edge_weights': edge_weights[high_edge_mask],
        'high_edge_indices': np.where(high_edge_mask)[0],
        'high_vertex_points': points[high_vertex_mask],
        'high_vertex_weights': vertex_weights[high_vertex_mask],
        'high_vertex_indices': np.where(high_vertex_mask)[0],
        'edge_count': np.sum(high_edge_mask),
        'vertex_count': np.sum(high_vertex_mask)
    }
    
    print(f"\n=== High Weight Point Summary ===")
    print(f"High edge points (>{edge_threshold}): {result['edge_count']}")
    print(f"High vertex points (>{vertex_threshold}): {result['vertex_count']}")
    
    return result
