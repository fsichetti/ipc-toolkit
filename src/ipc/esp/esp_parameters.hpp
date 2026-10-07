#pragma once

#include <ipc/barrier/barrier.hpp>
#include <ipc/utils/logger.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ipc {

/// A single face quadrature point in barycentric coordinates with its weight.
struct FaceQuadPoint {
    std::array<double, 3> lambda; ///< Barycentric coordinates (sum = 1)
    double weight;
};
using FaceQuadRule = std::vector<FaceQuadPoint>;

struct ESPParameters {
    enum class IntegrationType : std::uint8_t {
        BRUTE_FORCE, ///< Integrate all pairs with no obstacle filtering
        NORMAL, ///< Filter obstacle-obstacle pairs; skip primitives with only
                ///< obstacle candidates
        NO_OBST ///< Skip obstacle sources entirely, may miss collisions!
    };

    /// @param dbar_factor_value Near/far split parameter dbar / dhat, in
    ///        [0, 2]. The far part of the barrier vanishes identically at 2,
    ///        so the split tends continuously to the unsplit potential; the
    ///        edge-edge weights use the support min(dbar, dhat) (ee_support).
    ///        0 disables edge-edge contact.
    /// @throws std::invalid_argument if dbar_factor_value is not in [0, 2].
    ESPParameters(
        const double _dhat,
        const double dbar_factor_value = 0.2,
        const int _quad_order = 1,
        bool _area_weights = true,
        const IntegrationType _integration_type = IntegrationType::NORMAL)
        : dhat(_dhat)
        , dbar(dbar_factor_value * dhat)
        , dbar_factor_value(dbar_factor_value)
        , quad_order(_quad_order)
        , area_weights(_area_weights)
        , integration_type(_integration_type)
    {
        if (std::isnan(dbar_factor_value) || dbar_factor_value < 0
            || dbar_factor_value > 2) {
            throw std::invalid_argument(
                "dbar_factor " + std::to_string(dbar_factor_value)
                + " is not in [0, 2].");
        } else if (dbar_factor_value == 0) {
            logger().warn("dbar_factor = 0 disables edge-edge contact.");
        }
    }

    const double dhat;
    const double dbar;
    const double dbar_factor_value;

    /// Barrier evaluated on each pair's unsquared distance d with activation
    /// distance dhat, in 2D and 3D. With an InversePowerBarrier of power p, the
    /// 3D edge-edge mollifier weights are raised to the power max(1, round(p) +
    /// 1).
    std::shared_ptr<Barrier> barrier =
        std::make_shared<NormalizedClampedLogBarrier<>>();
    /// Gauss-Lobatto edge quadrature order (2D): order n uses n + 1 points and
    /// must be at least 1 in 2D. Unused in 3D.
    const int quad_order;
    /// Weight each face by its area / 9 (3D), or each edge by its area
    /// (CollisionMesh::edge_area(), 2D). If false, every weight is 1.
    bool area_weights;
    const IntegrationType integration_type;

    double dbar_factor() const { return dbar_factor_value; }

    /// @brief Support of the edge-edge (edge point) weights: min(dbar, dhat).
    /// Edge-edge pairs farther apart than this get no quadrature point.
    double ee_support() const { return std::min(dbar, dhat); }

    /// Face quadrature rule (3D). Empty (default) integrates over the face
    /// vertices instead; quad_order does not affect 3D.
    const FaceQuadRule& get_quad_rule() const { return m_face_quad_rule; }

    /// @brief Set the face quadrature rule (3D).
    /// @param rule Quadrature points in barycentric coordinates with their
    ///        weights. Empty integrates over the face vertices instead.
    /// @throws std::invalid_argument if a weight is negative or not finite, if
    ///         the weights sum to zero, or if a barycentric coordinate is
    ///         negative (point outside the face) or not finite. A negative
    ///         weight makes the barrier energy unbounded below as its point
    ///         approaches contact.
    /// @note Logs a warning for each point whose barycentric coordinates do
    ///       not sum to 1 (within 1e-10).
    void set_quad_rule(FaceQuadRule rule)
    {
        double weight_sum = 0;
        std::vector<size_t> not_affine; // points whose coordinates sum != 1
        for (size_t i = 0; i < rule.size(); i++) {
            const double weight = rule[i].weight;
            if (!std::isfinite(weight) || weight < 0) {
                throw std::invalid_argument(
                    fmt::format(
                        "Face quadrature point {} has negative or non-finite "
                        "weight {}.",
                        i, weight));
            }
            weight_sum += weight;

            double lambda_sum = 0;
            for (const double lambda : rule[i].lambda) {
                if (!std::isfinite(lambda) || lambda < 0) {
                    throw std::invalid_argument(
                        fmt::format(
                            "Face quadrature point {} has negative or "
                            "non-finite barycentric coordinate {}.",
                            i, lambda));
                }
                lambda_sum += lambda;
            }
            if (std::abs(lambda_sum - 1) > 1e-10) {
                not_affine.push_back(i);
            }
        }
        if (!rule.empty() && weight_sum == 0) {
            throw std::invalid_argument("Face quadrature weights sum to zero.");
        }
        for (const size_t i : not_affine) {
            const auto& l = rule[i].lambda;
            logger().warn(
                "Barycentric coordinates of face quadrature point {} sum to "
                "{}, not 1.",
                i, l[0] + l[1] + l[2]);
        }
        m_face_quad_rule = std::move(rule);
    }

    /// Record a distance passed to the barrier; tracks the running minimum
    /// across all threads. Copies of this struct share the same tracker
    /// (shared_ptr), so pass-by-value sites still update the original.
    void record_dist(double d) const
    {
        auto& a = *m_min_dist_seen;
        double cur = a.load(std::memory_order_relaxed);
        while (d < cur
               && !a.compare_exchange_weak(cur, d, std::memory_order_relaxed)) {
        }
    }

    double min_dist_seen() const
    {
        return m_min_dist_seen->load(std::memory_order_relaxed);
    }

    void reset_min_dist() const
    {
        m_min_dist_seen->store(
            std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
    }

private:
    FaceQuadRule m_face_quad_rule;

    std::shared_ptr<std::atomic<double>> m_min_dist_seen =
        std::make_shared<std::atomic<double>>(
            std::numeric_limits<double>::infinity());
};

} // namespace ipc
