#include "VBMicrolensingLibrary.h"

#include "xla/ffi/api/ffi.h"

#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>

namespace ffi = xla::ffi;
namespace py = pybind11;

namespace {

constexpr std::int64_t parameter_count = 5;
std::mutex vbm_mutex;

struct Jet {
    double value = 0.0;
    std::array<double, parameter_count> derivative{};

    Jet() = default;
    Jet(double scalar) : value(scalar) {}

    static Jet variable(double scalar, std::size_t index)
    {
        Jet result(scalar);
        result.derivative[index] = 1.0;
        return result;
    }
};

Jet operator+(const Jet& left, const Jet& right)
{
    Jet result(left.value + right.value);
    for (std::size_t index = 0; index < parameter_count; ++index) {
        result.derivative[index] =
            left.derivative[index] + right.derivative[index];
    }
    return result;
}

Jet operator-(const Jet& left, const Jet& right)
{
    Jet result(left.value - right.value);
    for (std::size_t index = 0; index < parameter_count; ++index) {
        result.derivative[index] =
            left.derivative[index] - right.derivative[index];
    }
    return result;
}

Jet operator-(const Jet& value)
{
    Jet result(-value.value);
    for (std::size_t index = 0; index < parameter_count; ++index) {
        result.derivative[index] = -value.derivative[index];
    }
    return result;
}

Jet operator*(const Jet& left, const Jet& right)
{
    Jet result(left.value * right.value);
    for (std::size_t index = 0; index < parameter_count; ++index) {
        result.derivative[index] =
            left.derivative[index] * right.value
            + left.value * right.derivative[index];
    }
    return result;
}

Jet operator/(const Jet& numerator, const Jet& denominator)
{
    Jet result(numerator.value / denominator.value);
    const double inverse_denominator_squared =
        1.0 / (denominator.value * denominator.value);
    for (std::size_t index = 0; index < parameter_count; ++index) {
        result.derivative[index] =
            (numerator.derivative[index] * denominator.value
             - numerator.value * denominator.derivative[index])
            * inverse_denominator_squared;
    }
    return result;
}

Jet jet_abs(const Jet& value)
{
    return value.value < 0.0 ? -value : value;
}

Jet jet_sqrt(const Jet& value)
{
    Jet result(std::sqrt(value.value));
    if (!(result.value > 0.0)) {
        result.derivative.fill(std::numeric_limits<double>::quiet_NaN());
        return result;
    }
    const double scale = 0.5 / result.value;
    for (std::size_t index = 0; index < parameter_count; ++index) {
        result.derivative[index] = scale * value.derivative[index];
    }
    return result;
}

struct ComplexJet {
    Jet real;
    Jet imag;
};

ComplexJet operator+(const ComplexJet& left, const ComplexJet& right)
{
    return {left.real + right.real, left.imag + right.imag};
}

ComplexJet operator*(const ComplexJet& left, const ComplexJet& right)
{
    return {
        left.real * right.real - left.imag * right.imag,
        left.real * right.imag + left.imag * right.real,
    };
}

ComplexJet operator*(const Jet& scalar, const ComplexJet& value)
{
    return {scalar * value.real, scalar * value.imag};
}

ComplexJet complex_inverse(const ComplexJet& value)
{
    const Jet norm = value.real * value.real + value.imag * value.imag;
    return {value.real / norm, -value.imag / norm};
}

ComplexJet complex_conjugate(const ComplexJet& value)
{
    return {value.real, -value.imag};
}

template <typename Scalar>
std::array<Scalar, 6> binary_lens_map_and_jacobian(
    const Scalar& image_x,
    const Scalar& image_y,
    const Scalar& separation,
    const Scalar& mass_ratio)
{
    const Scalar total_mass = 1.0 + mass_ratio;
    const Scalar mass_1 = 1.0 / total_mass;
    const Scalar mass_2 = mass_ratio / total_mass;
    const Scalar lens_1_x = -mass_2 * separation;
    const Scalar lens_2_x = mass_1 * separation;
    const Scalar dx_1 = image_x - lens_1_x;
    const Scalar dx_2 = image_x - lens_2_x;
    const Scalar y_squared = image_y * image_y;
    const Scalar radius_1_squared = dx_1 * dx_1 + y_squared;
    const Scalar radius_2_squared = dx_2 * dx_2 + y_squared;
    const Scalar inverse_1 = 1.0 / radius_1_squared;
    const Scalar inverse_2 = 1.0 / radius_2_squared;
    const Scalar source_x =
        image_x - mass_1 * dx_1 * inverse_1 - mass_2 * dx_2 * inverse_2;
    const Scalar source_y =
        image_y - mass_1 * image_y * inverse_1
        - mass_2 * image_y * inverse_2;
    const Scalar shear_real =
        mass_1 * (dx_1 * dx_1 - y_squared) * inverse_1 * inverse_1
        + mass_2 * (dx_2 * dx_2 - y_squared) * inverse_2 * inverse_2;
    const Scalar shear_cross =
        2.0 * image_y
        * (mass_1 * dx_1 * inverse_1 * inverse_1
           + mass_2 * dx_2 * inverse_2 * inverse_2);
    return {
        source_x,
        source_y,
        1.0 + shear_real,
        shear_cross,
        shear_cross,
        1.0 - shear_real,
    };
}

// Source position only -- the two components the implicit-function-theorem
// right-hand side needs at a fixed root.  Skips the shear that
// binary_lens_map_and_jacobian would also propagate.
template <typename Scalar>
std::array<Scalar, 2> binary_lens_source(
    const Scalar& image_x,
    const Scalar& image_y,
    const Scalar& separation,
    const Scalar& mass_ratio)
{
    const Scalar total_mass = 1.0 + mass_ratio;
    const Scalar mass_1 = 1.0 / total_mass;
    const Scalar mass_2 = mass_ratio / total_mass;
    const Scalar dx_1 = image_x + mass_2 * separation;
    const Scalar dx_2 = image_x - mass_1 * separation;
    const Scalar y_squared = image_y * image_y;
    const Scalar inverse_1 = 1.0 / (dx_1 * dx_1 + y_squared);
    const Scalar inverse_2 = 1.0 / (dx_2 * dx_2 + y_squared);
    return {
        image_x - mass_1 * dx_1 * inverse_1 - mass_2 * dx_2 * inverse_2,
        image_y - mass_1 * image_y * inverse_1 - mass_2 * image_y * inverse_2,
    };
}

// Real 2-by-2 lens Jacobian entries {1 + s_re, s_cross, s_cross, 1 - s_re}.
// Skips the source position that binary_lens_map_and_jacobian would also carry.
template <typename Scalar>
std::array<Scalar, 4> binary_lens_jacobian(
    const Scalar& image_x,
    const Scalar& image_y,
    const Scalar& separation,
    const Scalar& mass_ratio)
{
    const Scalar total_mass = 1.0 + mass_ratio;
    const Scalar mass_1 = 1.0 / total_mass;
    const Scalar mass_2 = mass_ratio / total_mass;
    const Scalar dx_1 = image_x + mass_2 * separation;
    const Scalar dx_2 = image_x - mass_1 * separation;
    const Scalar y_squared = image_y * image_y;
    const Scalar inverse_1 = 1.0 / (dx_1 * dx_1 + y_squared);
    const Scalar inverse_2 = 1.0 / (dx_2 * dx_2 + y_squared);
    const Scalar shear_real =
        mass_1 * (dx_1 * dx_1 - y_squared) * inverse_1 * inverse_1
        + mass_2 * (dx_2 * dx_2 - y_squared) * inverse_2 * inverse_2;
    const Scalar shear_cross =
        2.0 * image_y
        * (mass_1 * dx_1 * inverse_1 * inverse_1
           + mass_2 * dx_2 * inverse_2 * inverse_2);
    return {
        1.0 + shear_real,
        shear_cross,
        shear_cross,
        1.0 - shear_real,
    };
}

Jet analytic_point_magnification_from_images(
    const _sols_for_skiplist_curve* images,
    double source_x,
    double source_y,
    double q,
    double separation)
{
    const Jet active_separation = Jet::variable(separation, 4);
    const Jet active_q = Jet::variable(q, 3);
    Jet magnification(0.0);

    for (auto* curve = images->first; curve; curve = curve->next) {
        const auto* root = curve->first;
        if (root == nullptr) continue;
        // BinaryMag0 leaves point roots in VBM's internal lens-origin frame.
        // VBM swaps the labelled masses for q >= 1, which changes the COM
        // translation while leaving the physical lens map unchanged.
        const double root_x = q < 1.0
            ? root->x1 + separation / (1.0 + q)
            : root->x1 - separation * q / (1.0 + q);
        const double root_y = root->x2;
        const auto mapped = binary_lens_map_and_jacobian(
            root_x, root_y, separation, q);
        const double determinant =
            mapped[2] * mapped[5] - mapped[3] * mapped[4];
        if (!(std::abs(determinant) > 1.0e-12)) continue;

        const auto fixed_root_source = binary_lens_source(
            Jet(root_x), Jet(root_y), active_separation, active_q);
        Jet active_root_x(root_x);
        Jet active_root_y(root_y);
        for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
            const double source_x_tangent = parameter == 0 ? 1.0 : 0.0;
            const double source_y_tangent =
                parameter == 1
                ? (source_y < 0.0 ? -1.0 : (source_y > 0.0 ? 1.0 : 0.0))
                : 0.0;
            const double rhs_x =
                source_x_tangent - fixed_root_source[0].derivative[parameter];
            const double rhs_y =
                source_y_tangent - fixed_root_source[1].derivative[parameter];
            active_root_x.derivative[parameter] =
                (mapped[5] * rhs_x - mapped[3] * rhs_y) / determinant;
            active_root_y.derivative[parameter] =
                (-mapped[4] * rhs_x + mapped[2] * rhs_y) / determinant;
        }
        const auto active_jacobian = binary_lens_jacobian(
            active_root_x, active_root_y, active_separation, active_q);
        const Jet active_determinant =
            active_jacobian[0] * active_jacobian[3]
            - active_jacobian[1] * active_jacobian[2];
        magnification = magnification
            + Jet(1.0) / (determinant >= 0.0
                              ? active_determinant
                              : Jet(-1.0) * active_determinant);
    }
    return magnification;
}

bool contour_has_caustic_connections(const _sols_for_skiplist_curve* images)
{
    for (auto* curve = images->first; curve; curve = curve->next) {
        if (curve->partneratstart || curve->partneratend) return true;
    }
    return false;
}

struct ActiveImagePoint {
    Jet x;
    Jet y;
    Jet dx_dtheta;
    Jet dy_dtheta;
    // ds = (rho^2 + Im(dy/dtheta * (dz/dtheta)^2 * d2y/dz2)) / dJ.  The ds-form
    // parabolic-correction coefficient (VBM's scan->ds).  Feeds both the
    // ordinary segment's ds-form parabola / gradient-error estimator and the
    // caustic-connection wedge acceleration.
    Jet ds;
    double theta;
};

ActiveImagePoint differentiate_finite_image_point(
    const _point* point,
    double source_x,
    double reflected_source_y,
    double rho,
    double q,
    double separation)
{
    const double root_x = point->x1;
    const double root_y = point->x2;
    const auto mapped = binary_lens_map_and_jacobian(
        root_x, root_y, separation, q);
    const double determinant = mapped[2] * mapped[5] - mapped[3] * mapped[4];

    const Jet active_source_x = Jet::variable(source_x, 0);
    const Jet active_source_y = Jet::variable(reflected_source_y, 1);
    const Jet active_rho = Jet::variable(rho, 2);
    const Jet active_q = Jet::variable(q, 3);
    const Jet active_separation = Jet::variable(separation, 4);
    const double boundary_unit_x = (mapped[0] - source_x) / rho;
    const double boundary_unit_y = (mapped[1] - reflected_source_y) / rho;
    const Jet boundary_x = active_source_x + active_rho * boundary_unit_x;
    const Jet boundary_y = active_source_y + active_rho * boundary_unit_y;
    // The source-limb angle theta_i for this root is a parameter-independent
    // constant, so the implicit-function derivatives below are on a frozen
    // mesh and carry no mesh-motion term.
    const auto fixed_root_source = binary_lens_source(
        Jet(root_x), Jet(root_y), active_separation, active_q);

    ActiveImagePoint result{
        Jet(root_x), Jet(root_y), Jet(0.0), Jet(0.0), Jet(0.0),
        std::atan2(boundary_unit_y, boundary_unit_x)};
    for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
        const double rhs_x = boundary_x.derivative[parameter]
            - fixed_root_source[0].derivative[parameter];
        const double rhs_y = boundary_y.derivative[parameter]
            - fixed_root_source[1].derivative[parameter];
        result.x.derivative[parameter] =
            (mapped[5] * rhs_x - mapped[3] * rhs_y) / determinant;
        result.y.derivative[parameter] =
            (-mapped[4] * rhs_x + mapped[2] * rhs_y) / determinant;
    }

    // One complex evaluation at the differentiated root.  The first lens
    // derivative gives the 2-by-2 Jacobian entries (shear_real = Re J1,
    // shear_cross = Im J1) and its determinant dJ = 1 - |J1|^2; the second lens
    // derivative gives ds.  Sharing delta_1/delta_2 and their inverses keeps
    // this close to the cost of the plain Jacobian evaluation it replaces.
    const Jet total_mass = 1.0 + active_q;
    const Jet mass_1 = 1.0 / total_mass;
    const Jet mass_2 = active_q / total_mass;
    const ComplexJet delta_1{result.x + mass_2 * active_separation, result.y};
    const ComplexJet delta_2{result.x - mass_1 * active_separation, result.y};
    const ComplexJet inverse_1 = complex_inverse(delta_1);
    const ComplexJet inverse_2 = complex_inverse(delta_2);
    const ComplexJet inverse_conjugate_1 = complex_conjugate(inverse_1);
    const ComplexJet inverse_conjugate_2 = complex_conjugate(inverse_2);
    const ComplexJet lens_first_derivative =
        mass_1 * (inverse_conjugate_1 * inverse_conjugate_1)
        + mass_2 * (inverse_conjugate_2 * inverse_conjugate_2);
    const Jet& shear_real = lens_first_derivative.real;
    const Jet& shear_cross = lens_first_derivative.imag;
    const Jet active_determinant =
        Jet(1.0) - shear_real * shear_real - shear_cross * shear_cross;

    const Jet tangent_x = -active_rho * boundary_unit_y;
    const Jet tangent_y = active_rho * boundary_unit_x;
    result.dx_dtheta =
        ((Jet(1.0) - shear_real) * tangent_x - shear_cross * tangent_y)
        / active_determinant;
    result.dy_dtheta =
        (-shear_cross * tangent_x + (Jet(1.0) + shear_real) * tangent_y)
        / active_determinant;

    const ComplexJet lens_second_derivative = Jet(-2.0)
        * (mass_1 * (inverse_1 * inverse_1 * inverse_1)
           + mass_2 * (inverse_2 * inverse_2 * inverse_2));
    const ComplexJet image_tangent{result.dx_dtheta, result.dy_dtheta};
    const ComplexJet source_tangent{
        -active_rho * std::sin(result.theta),
        active_rho * std::cos(result.theta)};
    const ComplexJet curvature_product =
        image_tangent * image_tangent * source_tangent
        * lens_second_derivative;
    result.ds =
        (active_rho * active_rho + curvature_product.imag) / active_determinant;
    return result;
}

// Ordinary (non-caustic) segment parabolic correction and its analytic
// gradient-error contribution.
//
// VBM forms the segment correction as 0.5 * (parab1 + parab2) of two
// independent fourth-order estimates at the *same* contour resolution:
//   parab1  (ds form)      = dtheta^3 / 24 * (ds_i + ds_{i+1})
//   parab2  (tangent form) = 1/12 (dx dd.im - dy dd.re) dtheta
// and refines the primal value on their disagreement |parab1 - parab2| / 2.
// The parameter derivative of that same disagreement,
//   0.5 * |d(parab1)/dp - d(parab2)/dp|,
// is the analytic Jacobian-error estimator for parameter p -- no resolution
// sweep, no finite differencing.  It is accumulated for the four support
// parameters (source_x, source_y, q, separation); the rho column is derived
// separately in finite_contour_magnification because d(parab)/d(rho) is
// dominated by the exact d(rho^2)/d(rho) term of the ds form and its raw
// two-form disagreement carries no error signal.
Jet ordinary_parabolic_correction(
    const _point* point,
    const ActiveImagePoint& active_point,
    const ActiveImagePoint& active_next,
    std::array<double, parameter_count>& derivative_error)
{
    double delta_theta = active_point.theta - active_next.theta;
    if (delta_theta >= 0.0) delta_theta -= 2.0 * M_PI;
    const double delta_theta_squared = delta_theta * delta_theta;

    const Jet parab_tangent = (1.0 / 12.0)
        * ((active_next.x - active_point.x)
               * (active_next.dy_dtheta - active_point.dy_dtheta)
           - (active_next.y - active_point.y)
               * (active_next.dx_dtheta - active_point.dx_dtheta))
        * delta_theta;
    const Jet parab_ds =
        (delta_theta_squared * delta_theta / 24.0)
        * (active_point.ds + active_next.ds);
    const Jet combined = 0.5 * (parab_ds + parab_tangent);

    // Geometric vs angular spacing (VBM's cmp2 / cmp_2): the segment length
    // over the tangent dot-product is a "distance" estimate of dtheta^2; its
    // mismatch with the true dtheta^2 penalises a distorted mesh.
    const double segment_length_squared =
        (active_next.x.value - active_point.x.value)
            * (active_next.x.value - active_point.x.value)
        + (active_next.y.value - active_point.y.value)
            * (active_next.y.value - active_point.y.value);
    const double tangent_dot = std::abs(
        active_point.dx_dtheta.value * active_next.dx_dtheta.value
        + active_point.dy_dtheta.value * active_next.dy_dtheta.value);
    const double spacing_squared = tangent_dot > 0.0
        ? segment_length_squared / tangent_dot
        : delta_theta_squared;
    const double mesh_penalty = delta_theta_squared > 0.0
        ? spacing_squared * 0.1
              + 1.5 * std::abs(spacing_squared / delta_theta_squared - 1.0)
        : 0.0;

    // Use the two-form disagreement only when the reconstructed correction
    // lands on VBM's stored value; otherwise fall back to the proven
    // scaled-tangent derivative with a conservative error bound.
    const bool reconstruction_ok =
        std::isfinite(combined.value)
        && (point->parab == 0.0
                ? std::abs(combined.value) < 1.0e-12
                : std::abs(combined.value - point->parab)
                      <= 0.5 * std::abs(point->parab) + 1.0e-15);

    if (reconstruction_ok) {
        // Accumulate the per-parameter two-form derivative disagreement.
        // The rho column (index 2) is discarded here and rebuilt from the
        // support columns in finite_contour_magnification.
        for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
            if (parameter == 2) continue;
            derivative_error[parameter] += 0.5
                * std::abs(parab_ds.derivative[parameter]
                           - parab_tangent.derivative[parameter]);
        }
        return Jet(point->parab) + combined - Jet(combined.value);
    }

    double derivative_scale = 1.0;
    if (std::abs(parab_tangent.value) > 1.0e-30) {
        const double ratio = point->parab / parab_tangent.value;
        if (ratio > 0.5 && ratio < 1.5) derivative_scale = ratio;
    }
    const Jet scaled = Jet(point->parab)
        + derivative_scale * (parab_tangent - Jet(parab_tangent.value));
    for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
        if (parameter == 2) continue;
        derivative_error[parameter] +=
            std::abs(scaled.derivative[parameter])
            * std::max(delta_theta_squared, mesh_penalty);
    }
    return scaled;
}

Jet critical_parabolic_correction(
    double stored_parabola,
    const ActiveImagePoint& first,
    const ActiveImagePoint& second,
    bool creation,
    std::array<double, parameter_count>& derivative_error)
{
    const Jet delta_x = second.x - first.x;
    const Jet delta_y = second.y - first.y;
    const Jet distance_squared = delta_x * delta_x + delta_y * delta_y;
    const Jet tangent_dot =
        first.dx_dtheta * second.dx_dtheta
        + first.dy_dtheta * second.dy_dtheta;
    const Jet delta_theta_tilde =
        jet_sqrt(distance_squared / jet_abs(tangent_dot));
    const Jet wedge =
        delta_x * (second.dy_dtheta + first.dy_dtheta)
        - delta_y * (second.dx_dtheta + first.dx_dtheta);
    const Jet curvature_parabola =
        (creation ? 1.0 / 24.0 : -1.0 / 24.0)
        * (first.ds - second.ds)
        * delta_theta_tilde * delta_theta_tilde * delta_theta_tilde;
    const Jet tangent_parabola =
        (creation ? -1.0 / 12.0 : 1.0 / 12.0)
        * wedge * delta_theta_tilde;
    const Jet active_parabola =
        0.5 * (curvature_parabola + tangent_parabola);
    for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
        derivative_error[parameter] += 0.5 * std::abs(
            curvature_parabola.derivative[parameter]
            - tangent_parabola.derivative[parameter]);
    }
    return Jet(stored_parabola)
        + active_parabola - Jet(active_parabola.value);
}

struct ContourDerivativeResult {
    Jet magnification;
    // Analytic per-parameter Jacobian-error estimate for this contour support:
    // the summed two-form parabola disagreement (ordinary segments) plus the
    // creation/destruction curvature-vs-tangent disagreement (caustic
    // connections), scaled to d(magnification)/d(parameter) units.
    std::array<double, parameter_count> derivative_error{};
};

ContourDerivativeResult finite_contour_magnification(
    const _sols_for_skiplist_curve* images,
    double source_x,
    double source_y,
    double rho,
    double q,
    double separation)
{
    const double reflected_source_y = std::abs(source_y);
    const double reflection_tangent =
        source_y < 0.0 ? -1.0 : (source_y > 0.0 ? 1.0 : 0.0);
    Jet area(0.0);
    std::array<double, parameter_count> derivative_error{};
    for (auto* curve = images->first; curve; curve = curve->next) {
        ActiveImagePoint active_point = differentiate_finite_image_point(
            curve->first, source_x, reflected_source_y, rho, q, separation);
        for (auto* point = curve->first; point && point->next; point = point->next) {
            const auto active_next = differentiate_finite_image_point(
                point->next, source_x, reflected_source_y, rho, q, separation);
            const Jet trapezoid = 0.5 * (active_point.y + active_next.y)
                * (active_next.x - active_point.x);
            const Jet parabola = ordinary_parabolic_correction(
                point, active_point, active_next, derivative_error);
            const double orientation = point->dJ > 0.0 ? -1.0 : 1.0;
            area = area + orientation * (trapezoid + parabola);
            active_point = active_next;
        }
        const auto ordered_before = std::less<const _skiplist_curve*>{};
        if (
            curve->partneratstart
            && ordered_before(curve, curve->partneratstart)) {
            const auto first = differentiate_finite_image_point(
                curve->first, source_x, reflected_source_y, rho, q, separation);
            const auto second = differentiate_finite_image_point(
                curve->partneratstart->first,
                source_x, reflected_source_y, rho, q, separation);
            const Jet trapezoid =
                0.5 * (first.y + second.y)
                * (second.x - first.x);
            const Jet parabola = critical_parabolic_correction(
                curve->parabstart, first, second, true, derivative_error);
            const double orientation = curve->first->dJ > 0.0 ? -1.0 : 1.0;
            area = area - orientation * (trapezoid + parabola);
        }
        if (
            curve->partneratend
            && ordered_before(curve, curve->partneratend)) {
            const auto first = differentiate_finite_image_point(
                curve->last, source_x, reflected_source_y, rho, q, separation);
            const auto second = differentiate_finite_image_point(
                curve->partneratend->last,
                source_x, reflected_source_y, rho, q, separation);
            const Jet trapezoid =
                0.5 * (first.y + second.y)
                * (second.x - first.x);
            const Jet parabola = critical_parabolic_correction(
                curve->last->parab, first, second, false, derivative_error);
            const double orientation = curve->last->dJ > 0.0 ? -1.0 : 1.0;
            area = area + orientation * (trapezoid + parabola);
        }
    }
    area.derivative[1] *= reflection_tangent;
    derivative_error[1] *= std::abs(reflection_tangent);
    // The rho column: d(parab)/d(rho) of the ds form is dominated by the exact
    // d(rho^2)/d(rho) = 2*rho term (rho^2 is the geometric identity |dy/dtheta|^2,
    // not an approximation), which the tangent form expresses only implicitly, so
    // the raw two-form disagreement is a ~1e3x pessimistic non-signal.  The
    // frozen-support Jacobian error in d(mag)/d(rho) instead tracks the aggregate
    // support-placement error already visible in the four support columns; bound
    // it by their sum.
    derivative_error[2] = derivative_error[0] + derivative_error[1]
        + derivative_error[3] + derivative_error[4];
    const Jet active_rho = Jet::variable(rho, 2);
    const Jet magnification = area / (M_PI * active_rho * active_rho);
    const double inverse_source_area = 1.0 / (M_PI * rho * rho);
    for (double& error : derivative_error) error *= inverse_source_area;
    return {magnification, derivative_error};
}

// Dimensionless per-epoch Jacobian-error estimate: the analytic per-parameter
// error, relative to the derivative magnitude where that exceeds unity.
double contour_derivative_error(const ContourDerivativeResult& result)
{
    double error = 0.0;
    for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
        error = std::max(
            error,
            result.derivative_error[parameter]
                / std::max(
                    1.0,
                    std::abs(result.magnification.derivative[parameter])));
    }
    return error;
}

bool jet_is_finite(const Jet& value)
{
    if (!std::isfinite(value.value)) return false;
    return std::all_of(
        value.derivative.begin(), value.derivative.end(),
        [](double derivative) { return std::isfinite(derivative); });
}

double jet_derivative_change(const Jet& current, const Jet& previous)
{
    double change = 0.0;
    for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
        const double scale = std::max(1.0, std::abs(current.derivative[parameter]));
        change = std::max(
            change,
            std::abs(current.derivative[parameter] - previous.derivative[parameter])
                / scale);
    }
    return change;
}

ffi::Error validate_inputs(
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR1<ffi::F64> separation,
    ffi::BufferR1<ffi::S32> branch_mode,
    ffi::ResultBufferR1<ffi::F64>& magnification)
{
    const auto size = source_x.dimensions()[0];
    if (
        source_y.dimensions()[0] != size
        || separation.dimensions()[0] != size
        || branch_mode.dimensions()[0] != size
        || magnification->dimensions()[0] != size) {
        return ffi::Error::InvalidArgument("source, branch mode, separation, and magnification arrays must have equal length");
    }
    return ffi::Error::Success();
}

double evaluate_epoch(
    VBMicrolensing& vbm,
    double source_x,
    double source_y,
    double rho,
    double q,
    double separation,
    std::int32_t branch_mode)
{
    // VBMicrolensing's public BinaryMag2 interface expects the source in the
    // centre-of-mass frame.  The native FFI uses that same convention as the
    // public pure-JAX binary_mag/binary_mag_from_trajectory APIs.
    //
    // For the deterministic HMC branches (0 / 1) the forward value must be the
    // SAME effective function the Jacobian handler differentiates: VBM's
    // point/contour solve fixes the image support, and the magnification is
    // then reconstructed from that frozen support by the identical area walk
    // used for the analytic Jacobian (analytic_point_magnification_from_images
    // / finite_contour_magnification).  Returning VBM's own adaptively-refined
    // BinaryMag/BinaryMag0 value here instead would leave d(value)/d(theta)
    // inconsistent with the reported Jacobian near caustics -- the HMC
    // Hamiltonian-error source this backend exists to avoid.  VBM's value is
    // kept only as a fallback when the reconstruction is not usable (in which
    // case the Jacobian row is NaN anyway).
    if (branch_mode == 0) {
        _sols_for_skiplist_curve* images = nullptr;
        const double vbm_value = vbm.BinaryMag0(
            separation, q, source_x, std::abs(source_y), &images);
        const Jet point = analytic_point_magnification_from_images(
            images, source_x, source_y, q, separation);
        delete images;
        return std::isfinite(point.value) ? point.value : vbm_value;
    }
    if (branch_mode == 1) {
        _sols_for_skiplist_curve* images = nullptr;
        const double vbm_value = vbm.BinaryMag(
            separation, q, source_x, std::abs(source_y), rho, vbm.Tol, &images);
        const ContourDerivativeResult contour = finite_contour_magnification(
            images, source_x, source_y, rho, q, separation);
        delete images;
        const double reconstructed = contour.magnification.value;
        return (std::isfinite(reconstructed) && reconstructed > 0.0)
            ? reconstructed
            : vbm_value;
    }
    return vbm.BinaryMag2(separation, q, source_x, source_y, rho);
}

ffi::Error vbm_batch_ffi_impl(
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> rho,
    ffi::BufferR0<ffi::F64> q,
    ffi::BufferR1<ffi::F64> separation,
    ffi::BufferR1<ffi::S32> branch_mode,
    ffi::BufferR0<ffi::F64> tolerance,
    ffi::BufferR0<ffi::F64> relative_tolerance,
    ffi::ResultBufferR1<ffi::F64> magnification)
{
    const std::lock_guard<std::mutex> lock(vbm_mutex);
    auto validation = validate_inputs(
        source_x, source_y, separation, branch_mode, magnification);
    if (validation.failure()) return validation;

    VBMicrolensing vbm;
    vbm.Tol = *tolerance.typed_data();
    vbm.RelTol = *relative_tolerance.typed_data();
    const auto size = source_x.dimensions()[0];
    for (std::int64_t index = 0; index < size; ++index) {
        magnification->typed_data()[index] = evaluate_epoch(
            vbm,
            source_x.typed_data()[index],
            source_y.typed_data()[index],
            *rho.typed_data(),
            *q.typed_data(),
            separation.typed_data()[index],
            branch_mode.typed_data()[index]);
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    vbm_batch_ffi_handler,
    vbm_batch_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>());

ffi::Error vbm_batch_jacobian_ffi_impl(
    ffi::BufferR1<ffi::F64> source_x,
    ffi::BufferR1<ffi::F64> source_y,
    ffi::BufferR0<ffi::F64> rho,
    ffi::BufferR0<ffi::F64> q,
    ffi::BufferR1<ffi::F64> separation,
    ffi::BufferR1<ffi::S32> branch_mode,
    ffi::BufferR0<ffi::F64> tolerance,
    ffi::BufferR0<ffi::F64> relative_tolerance,
    ffi::BufferR0<ffi::F64> gradient_tolerance,
    ffi::BufferR0<ffi::S32> strict_grad_gate,
    ffi::ResultBufferR1<ffi::F64> magnification,
    ffi::ResultBufferR2<ffi::F64> jacobian,
    ffi::ResultBufferR1<ffi::F64> gradient_error)
{
    const std::lock_guard<std::mutex> lock(vbm_mutex);
    auto validation = validate_inputs(
        source_x, source_y, separation, branch_mode, magnification);
    if (validation.failure()) return validation;
    const auto size = source_x.dimensions()[0];
    const auto jacobian_dimensions = jacobian->dimensions();
    if (
        jacobian_dimensions[0] != size
        || jacobian_dimensions[1] != parameter_count) {
        return ffi::Error::InvalidArgument(
            "VBM Jacobian must have shape (epochs, 5)");
    }
    if (gradient_error->dimensions()[0] != size) {
        return ffi::Error::InvalidArgument(
            "VBM gradient-error map must have length epochs");
    }

    VBMicrolensing vbm;
    vbm.Tol = *tolerance.typed_data();
    vbm.RelTol = *relative_tolerance.typed_data();
    const double base_relative_tolerance = vbm.RelTol;

    // Per-epoch Jacobian-error budget for the analytic two-form estimator.  An
    // explicit grad_tol is honoured verbatim (HMC-grade gradients: pass e.g.
    // 3e-4).  A non-positive request falls back to a value-tied default: the
    // frozen-support Jacobian is checked to ~10x the magnification tolerance,
    // loose enough that ordinary epochs pass without a refinement solve yet
    // tight enough to NaN-gate a genuinely unconverged epoch.  The estimator is
    // a conservative bound, so a small value certifies the Jacobian outright; a
    // large one is resolved by the refinement loop below.
    double grad_budget = *gradient_tolerance.typed_data();
    if (!(grad_budget > 0.0)) {
        grad_budget = 10.0 * vbm.Tol;
        if (grad_budget < 1.0e-4) grad_budget = 1.0e-4;
        if (grad_budget > 1.0e-1) grad_budget = 1.0e-1;
    }

    const double rho_value = *rho.typed_data();
    const double q_value = *q.typed_data();
    // Default (false): an epoch whose analytic Jacobian-error estimator does not
    // certify within grad_budget still returns its best-effort finite Jacobian
    // row (the last refinement iterate).  This keeps HMC leapfrog momenta finite
    // -- a slightly-off derivative is a smaller lie than a NaN row that poisons
    // the whole GP gradient.  Opt in to strict NaN-gating for offline diagnostics
    // that want an explicit "uncertified" signal.
    const bool strict = *strict_grad_gate.typed_data() != 0;
    for (std::int64_t index = 0; index < size; ++index) {
        vbm.RelTol = base_relative_tolerance;
        const std::int32_t branch = branch_mode.typed_data()[index];
        const bool force_point = branch == 0;
        const bool force_finite = branch == 1;
        const std::array<double, parameter_count> parameters = {
            source_x.typed_data()[index],
            source_y.typed_data()[index],
            rho_value,
            q_value,
            separation.typed_data()[index],
        };
        _sols_for_skiplist_curve* point_images = nullptr;
        double point_magnification = 0.0;
        if (!force_finite) {
            point_magnification = vbm.BinaryMag0(
                parameters[4], parameters[3], parameters[0],
                std::abs(parameters[1]), &point_images);
        }
        if (force_point) {
            const Jet point = analytic_point_magnification_from_images(
                point_images,
                parameters[0],
                parameters[1],
                parameters[3],
                parameters[4]);
            delete point_images;
            // Report the reconstruction's own value so it is exactly the
            // function whose derivative this row carries.
            magnification->typed_data()[index] =
                std::isfinite(point.value) ? point.value : point_magnification;
            for (
                std::int64_t parameter = 0;
                parameter < parameter_count;
                ++parameter) {
                jacobian->typed_data()[index * parameter_count + parameter] =
                    point.derivative[parameter];
            }
            gradient_error->typed_data()[index] = 0.0;
            continue;
        }
        const double rho_squared = parameters[2] * parameters[2];
        const bool finite_source = force_finite || !(
            vbm.corrquad * 6.0 * (rho_squared + 1.0e-4 * vbm.Tol) < vbm.Tol
            && vbm.corrquad2 * 256.0 * (rho_squared + 1.0e-8) < 1.0
            && vbm.safedist > 4.0 * rho_squared);

        if (!finite_source) {
            const Jet point = analytic_point_magnification_from_images(
                point_images,
                parameters[0],
                parameters[1],
                parameters[3],
                parameters[4]);
            delete point_images;
            magnification->typed_data()[index] =
                std::isfinite(point.value) ? point.value : point_magnification;
            for (
                std::int64_t parameter = 0;
                parameter < parameter_count;
                ++parameter) {
                jacobian->typed_data()[index * parameter_count + parameter] =
                    point.derivative[parameter];
            }
            gradient_error->typed_data()[index] = 0.0;
            continue;
        }
        delete point_images;

        if (finite_source) {
            _sols_for_skiplist_curve* images = nullptr;
            const double value_tol = vbm.Tol;
            magnification->typed_data()[index] = vbm.BinaryMag(
                parameters[4], parameters[3], parameters[0],
                std::abs(parameters[1]), parameters[2], value_tol, &images);
            const bool has_caustic_connections =
                contour_has_caustic_connections(images);
            Jet finite;
            bool converged = false;
            double achieved_gradient_error =
                std::numeric_limits<double>::infinity();
            if (!has_caustic_connections) {
                // An ordinary finite-source epoch takes its Jacobian from VBM's
                // value-error contour support and checks it with the analytic
                // two-form parabola-disagreement estimator (same mathematical
                // basis as VBM's own |parab1 - parab2| value estimator, zero
                // extra contour solves).  Refine only when the estimator says
                // the support's Jacobian misses grad_budget -- no resolution
                // sweep, no finite differencing.  On events like MB21333 the
                // primal support already passes and the loop below never runs.
                ContourDerivativeResult ordinary = finite_contour_magnification(
                    images,
                    parameters[0], parameters[1], parameters[2], parameters[3],
                    parameters[4]);
                finite = ordinary.magnification;
                double estimator = contour_derivative_error(ordinary);
                achieved_gradient_error = estimator;
                converged = jet_is_finite(finite) && estimator <= grad_budget;

                double refine_tol = value_tol;
                double previous_estimator = estimator;
                for (int level = 0; level < 5 && !converged; ++level) {
                    refine_tol *= 0.25;
                    if (refine_tol < 1.0e-12) break;
                    _sols_for_skiplist_curve* refined_images = nullptr;
                    const double saved = vbm.RelTol;
                    vbm.RelTol = refine_tol;
                    vbm.BinaryMag(
                        parameters[4], parameters[3], parameters[0],
                        std::abs(parameters[1]), parameters[2], refine_tol,
                        &refined_images);
                    vbm.RelTol = saved;
                    ContourDerivativeResult refined = finite_contour_magnification(
                        refined_images,
                        parameters[0], parameters[1], parameters[2],
                        parameters[3], parameters[4]);
                    delete refined_images;
                    if (!jet_is_finite(refined.magnification)) break;
                    finite = refined.magnification;
                    estimator = contour_derivative_error(refined);
                    achieved_gradient_error = estimator;
                    if (estimator <= grad_budget) {
                        converged = true;
                        break;
                    }
                    // Early-out: a 4x contour refinement that barely moves the
                    // estimator is fighting a structural frozen-support error
                    // (a missed image feature), not a resolution shortfall --
                    // more solves will not reach the budget.  Refuse now
                    // instead of paying for the deep sweep.
                    if (previous_estimator < 1.5 * estimator) break;
                    previous_estimator = estimator;
                }
            }

            // The primal (value_tol) contour support, evaluated once for the
            // caustic branch below so a mild topology-change epoch whose frozen
            // support already carries an HMC-grade Jacobian can skip the
            // expensive resolution sweep entirely.
            Jet caustic_primal;
            double caustic_primal_estimator =
                std::numeric_limits<double>::infinity();
            if (has_caustic_connections) {
                const ContourDerivativeResult primal =
                    finite_contour_magnification(
                        images,
                        parameters[0], parameters[1], parameters[2],
                        parameters[3], parameters[4]);
                caustic_primal = primal.magnification;
                caustic_primal_estimator = contour_derivative_error(primal);
            }
            delete images;

            if (has_caustic_connections) {
                // Topology-changing epochs: VBM creates or destroys an image
                // curve inside the source disk, so the frozen contour support
                // omits the moving connection points.  finite_contour_magnification
                // still forms the creation/destruction wedge parabola and its
                // two-form disagreement, but a same-resolution estimate can be
                // fooled by a contour too coarse to resolve the caustic feature
                // at all.  Guard against that with a short, bounded resolution
                // sweep: certify the Jacobian only once the contour is fine
                // enough to see the wedge (<= 1e-5) *and* two successive
                // resolutions agree on the derivative *and* the parabola
                // estimator passes.  Otherwise leave the epoch NaN -- an honest
                // refusal is safer for HMC than a confidently wrong gradient.
                //
                // The sweep is deliberately shallow (four solves, floor 1e-7):
                // an HMC-grade Jacobian needs ~1e-3 relative accuracy, and a
                // contour certified at 1e-5..1e-6 already delivers that.  The
                // earlier design swept to 1e-12, which cost ~30 ms per caustic
                // epoch -- three such epochs dominated the whole batch.
                const double convergence_tolerance =
                    grad_budget < 1.0e-3 ? grad_budget : 1.0e-3;
                double requested_tolerance = vbm.Tol;
                if (
                    base_relative_tolerance > 0.0
                    && (requested_tolerance <= 0.0
                        || base_relative_tolerance < requested_tolerance)) {
                    requested_tolerance = base_relative_tolerance;
                }
                if (!(requested_tolerance > 0.0)) requested_tolerance = 1.0e-3;

                const double trust_tolerance = 1.0e-6;
                const double floor_tolerance = 1.0e-9;

                // When the primal support's Jacobian already clears the caustic
                // certification floor, start the sweep straight at the trust
                // tolerance and seed the stability check with the primal.  A
                // single confirmation solve then certifies the common
                // shallow-crossing epoch; only a support that fails outright
                // pays for the full 1e-4 -> 1e-9 ladder.
                const bool primal_certifies =
                    jet_is_finite(caustic_primal)
                    && caustic_primal_estimator <= convergence_tolerance;
                double contour_tolerance = primal_certifies
                    ? std::min(requested_tolerance, trust_tolerance)
                    : std::min(requested_tolerance, 1.0e-4);
                Jet previous = caustic_primal;
                bool have_previous = primal_certifies;
                for (int refinement = 0; refinement < 7; ++refinement) {
                    vbm.RelTol = contour_tolerance;
                    _sols_for_skiplist_curve* refined_images = nullptr;
                    vbm.BinaryMag(
                        parameters[4], parameters[3], parameters[0],
                        std::abs(parameters[1]), parameters[2],
                        contour_tolerance, &refined_images);
                    const ContourDerivativeResult refined_result =
                        finite_contour_magnification(
                        refined_images,
                        parameters[0], parameters[1], parameters[2],
                        parameters[3], parameters[4]);
                    delete refined_images;
                    const Jet& refined = refined_result.magnification;
                    if (!jet_is_finite(refined)) break;
                    finite = refined;
                    const double estimator =
                        contour_derivative_error(refined_result);
                    achieved_gradient_error = estimator;
                    if (have_previous) {
                        const double stability =
                            jet_derivative_change(refined, previous);
                        achieved_gradient_error = std::max(estimator, stability);
                        if (
                            contour_tolerance <= trust_tolerance
                            && estimator <= convergence_tolerance
                            && stability
                                   <= std::max(convergence_tolerance, 2.0e-3)) {
                            converged = true;
                            break;
                        }
                    }
                    previous = refined;
                    have_previous = true;
                    if (contour_tolerance <= floor_tolerance) break;
                    contour_tolerance = std::max(
                        contour_tolerance * 0.1, floor_tolerance);
                }
            }
            if (!converged && strict) {
                finite.derivative.fill(
                    std::numeric_limits<double>::quiet_NaN());
            }
            // Best-effort fallback (non-strict, the HMC default): if the caustic
            // sweep broke before ever assigning a finite reconstruction,
            // `finite` is still default-constructed -- substitute the primal
            // (value_tol) support Jet so the row is a real coarse Jacobian
            // rather than zeros.
            if (!strict && has_caustic_connections && !jet_is_finite(finite)
                && jet_is_finite(caustic_primal)) {
                finite = caustic_primal;
            }
            // Consistency: hand back the frozen-support reconstruction's own
            // value (the primal of the Jet whose derivative fills this row),
            // not VBM's separately-refined BinaryMag return.  finite.value is
            // untouched by the NaN-fill above.  Fall back to the VBM value only
            // if the reconstruction never produced a usable primal (caustic
            // sweep that broke before its first finite solve).
            if (std::isfinite(finite.value) && finite.value > 0.0) {
                magnification->typed_data()[index] = finite.value;
            }
            gradient_error->typed_data()[index] = achieved_gradient_error;
            for (
                std::int64_t parameter = 0;
                parameter < parameter_count;
                ++parameter) {
                jacobian->typed_data()[index * parameter_count + parameter] =
                    finite.derivative[parameter];
            }
            continue;
        }
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    vbm_batch_jacobian_ffi_handler,
    vbm_batch_jacobian_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<ffi::BufferR0<ffi::S32>>()   // strict_grad_gate
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>());

}  // namespace

PYBIND11_MODULE(_autodiff_vbm_native, module)
{
    module.doc() = "Fused VBMicrolensing CPU kernels for autodiff-vbm";
    module.def("vbm_batch_ffi", []() {
        return py::capsule(reinterpret_cast<void*>(vbm_batch_ffi_handler));
    });
    module.def("vbm_batch_jacobian_ffi", []() {
        return py::capsule(
            reinterpret_cast<void*>(vbm_batch_jacobian_ffi_handler));
    });
}
