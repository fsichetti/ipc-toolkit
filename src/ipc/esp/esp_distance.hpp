#pragma once

#include <ipc/distance/distance_type.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/line_line.hpp>
#include <ipc/distance/point_line.hpp>
#include <ipc/math/math.hpp>
#include <ipc/utils/autodiff_types.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/utils/logger.hpp>

#include <algorithm>

namespace ipc {

/// @brief Edge-edge squared distance for an EdgeEdgeDistanceType that may
/// come from edge_edge_distance_type_exact.
/// @note The exact classifier has no parallel threshold, so it can return
/// EA_EB for parallel edges, where the line-line distance divides by zero.
/// Those cases fall back to the minimum point-line distance.
inline double edge_edge_distance_parallel_safe(
    Eigen::ConstRef<Eigen::Vector3d> ea0,
    Eigen::ConstRef<Eigen::Vector3d> ea1,
    Eigen::ConstRef<Eigen::Vector3d> eb0,
    Eigen::ConstRef<Eigen::Vector3d> eb1,
    const EdgeEdgeDistanceType dtype)
{
    if (dtype == EdgeEdgeDistanceType::EA_EB) {
        const Eigen::Vector3d normal = (ea1 - ea0).cross(eb1 - eb0);
        if (normal.squaredNorm() > 1e-20) {
            return line_line_distance(ea0, ea1, eb0, eb1);
        }
        return std::min(
            { point_line_distance(eb0, ea0, ea1),
              point_line_distance(eb1, ea0, ea1),
              point_line_distance(ea0, eb0, eb1),
              point_line_distance(ea1, eb0, eb1) });
    }
    return edge_edge_distance(ea0, ea1, eb0, eb1, dtype);
}

template <typename T>
T point_point_sqr_distance(
    Eigen::ConstRef<Eigen::Vector3<T>> a, Eigen::ConstRef<Eigen::Vector3<T>> b)
{
    return (a - b).squaredNorm();
}

template <typename T>
T point_line_sqr_distance(
    Eigen::ConstRef<Eigen::Vector3<T>> p,
    Eigen::ConstRef<Eigen::Vector3<T>> e0,
    Eigen::ConstRef<Eigen::Vector3<T>> e1)
{
    return (e0 - p).cross(e1 - p).squaredNorm() / (e1 - e0).squaredNorm();
}

template <typename T>
T line_line_sqr_distance(
    Eigen::ConstRef<Eigen::Vector3<T>> ea0,
    Eigen::ConstRef<Eigen::Vector3<T>> ea1,
    Eigen::ConstRef<Eigen::Vector3<T>> eb0,
    Eigen::ConstRef<Eigen::Vector3<T>> eb1)
{
    const Eigen::Vector3<T> normal = (ea1 - ea0).cross(eb1 - eb0);
    const T line_to_line = (eb0 - ea0).dot(normal);
    return line_to_line * line_to_line / normal.squaredNorm();
}

template <typename scalar>
scalar edge_edge_sqr_distance(
    Eigen::ConstRef<Eigen::Vector3<scalar>> ea0,
    Eigen::ConstRef<Eigen::Vector3<scalar>> ea1,
    Eigen::ConstRef<Eigen::Vector3<scalar>> eb0,
    Eigen::ConstRef<Eigen::Vector3<scalar>> eb1,
    EdgeEdgeDistanceType dtype)
{
    if constexpr (std::is_same<double, scalar>::value) {
        if (dtype == EdgeEdgeDistanceType::AUTO) {
            dtype = edge_edge_distance_type(ea0, ea1, eb0, eb1);
        }
    }

    switch (dtype) {
    case EdgeEdgeDistanceType::EA0_EB0:
        return point_point_sqr_distance<scalar>(ea0, eb0);

    case EdgeEdgeDistanceType::EA0_EB1:
        return point_point_sqr_distance<scalar>(ea0, eb1);

    case EdgeEdgeDistanceType::EA1_EB0:
        return point_point_sqr_distance<scalar>(ea1, eb0);

    case EdgeEdgeDistanceType::EA1_EB1:
        return point_point_sqr_distance<scalar>(ea1, eb1);

    case EdgeEdgeDistanceType::EA_EB0:
        return point_line_sqr_distance<scalar>(eb0, ea0, ea1);

    case EdgeEdgeDistanceType::EA_EB1:
        return point_line_sqr_distance<scalar>(eb1, ea0, ea1);

    case EdgeEdgeDistanceType::EA0_EB:
        return point_line_sqr_distance<scalar>(ea0, eb0, eb1);

    case EdgeEdgeDistanceType::EA1_EB:
        return point_line_sqr_distance<scalar>(ea1, eb0, eb1);

    case EdgeEdgeDistanceType::EA_EB:
        return line_line_sqr_distance<scalar>(ea0, ea1, eb0, eb1);

    default:
        throw std::invalid_argument(
            "Invalid distance type for edge-edge distance!");
    }
}

template <typename T>
Eigen::Vector<T, 2> line_line_closest_point_pairs_uv(
    Eigen::ConstRef<Eigen::Vector3<T>> ea0,
    Eigen::ConstRef<Eigen::Vector3<T>> ea1,
    Eigen::ConstRef<Eigen::Vector3<T>> eb0,
    Eigen::ConstRef<Eigen::Vector3<T>> eb1)
{
    const Eigen::Vector3<T> u = ea1 - ea0;
    const Eigen::Vector3<T> v = eb1 - eb0;
    const Eigen::Vector3<T> w = ea0 - eb0;

    const T a = u.squaredNorm();
    const T b = u.dot(v);
    const T c = v.squaredNorm();
    const T d = u.dot(w);
    const T e = v.dot(w);

    const T sN = (b * e - c * d);
    const T tN = (a * e - b * d);
    const T fac = a * c - pow(b, 2);
    assert(fac > 0);

    return Eigen::Vector<T, 2>(sN, tN) / fac;
}

// Compute the closest point local coordinate on edge (e0, e1) with respect to
// edge (e2, e3) This function is written in a consistent way as the edge-edge
// distance type classification
template <typename T>
T closest_point_uv(
    Eigen::ConstRef<Eigen::Vector3<T>> e0,
    Eigen::ConstRef<Eigen::Vector3<T>> e1,
    Eigen::ConstRef<Eigen::Vector3<T>> e2,
    Eigen::ConstRef<Eigen::Vector3<T>> e3,
    EdgeEdgeDistanceType dtype)
{
    Eigen::Vector<T, 3> u = e1 - e0;
    Eigen::Vector<T, 3> v = e3 - e2;

    T uv(0.);
    if (dtype == EdgeEdgeDistanceType::EA_EB) {
        Eigen::Vector2<T> uvs =
            line_line_closest_point_pairs_uv<T>(e0, e1, e2, e3);

        uv = uvs(0);
    } else if (dtype == EdgeEdgeDistanceType::EA_EB0) {
        const T a = u.squaredNorm();
        const T d = u.dot(e0 - e2);
        uv = (-d) / a;
    } else if (dtype == EdgeEdgeDistanceType::EA_EB1) {
        const T a = u.squaredNorm();
        const T b = u.dot(v);
        const T d = u.dot(e0 - e2);
        uv = (-d + b) / a;
    } else {
        log_and_throw_error(
            "edge-edge dtype {} cannot handle!", static_cast<int>(dtype));
    }

    if (uv < 0.) {
        uv = 0.;
    } else if (uv > 1.) {
        uv = 1.;
    }

    return uv;
}

} // namespace ipc
