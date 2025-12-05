// simpson3d_integrator.hpp
#ifndef SIMPSON3D_INTEGRATOR_HPP
#define SIMPSON3D_INTEGRATOR_HPP

#include <cstddef>
#include <cuComplex.h>

class Simpson3DTiledIntegratorImpl;

/**
 * @brief Simpson 3D Integration using tiled approach
 *
 * This class implements 3D numerical integration using Simpson's rule
 * with a tiled processing approach to decrease GPU memory usage.
 */
class Simpson3DTiledIntegrator {
  private:
    Simpson3DTiledIntegratorImpl *pImpl; // Pointer to implementation

  public:
    /**
     * @brief Constructor
     * @param Nx Grid size in X direction
     * @param Ny Grid size in Y direction
     * @param tile_z Number of z-slices to process per tile (default: 32)
     */
    Simpson3DTiledIntegrator(long Nx, long Ny, long tile_z = 32);

    /**
     * @brief Destructor
     */
    ~Simpson3DTiledIntegrator();

    // Disable copy constructor and assignment operator
    Simpson3DTiledIntegrator(const Simpson3DTiledIntegrator &) = delete;
    Simpson3DTiledIntegrator &operator=(const Simpson3DTiledIntegrator &) = delete;

    /**
     * @brief Perform 3D integration using Simpson's rule (device memory input)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f Pointer to function values (DEVICE memory, may be padded)
     * @param Nx Logical number of points in X direction (for weighting)
     * @param Ny Number of points in Y direction
     * @param Nz Number of points in Z direction
     * @param f_Nx Actual X dimension of d_f (Nx for unpadded, Nx+2 for padded, defaults to Nx)
     * @return Integrated value
     */
    double integrateDevice(double hx, double hy, double hz, double *d_f, long Nx, long Ny, long Nz, long f_Nx = 0);

    /**
     * @brief Perform 3D integration using Simpson's rule (complex array cast to double)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f Pointer to function values (DEVICE memory) - complex array cast to double
     * @param Nx Number of points in X direction
     * @param Ny Number of points in Y direction
     * @param Nz Number of points in Z direction
     * @param f_Nx Actual X dimension of complex array before casting (Nx for unpadded, Nx+2 for padded, defaults to Nx)
     * @return Integrated value
     */
    double integrateDeviceComplex(double hx, double hy, double hz, double *d_f, long Nx, long Ny, long Nz, long f_Nx = 0);

    /**
     * @brief Perform 3D integration of |psi|^2 using Simpson's rule (computes |psi|^2 on-the-fly from complex array)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f Pointer to function values (DEVICE memory, complex array, unpadded, uses Nx for indexing)
     * @param Nx Number of points in X direction (also used for indexing)
     * @param Ny Number of points in Y direction
     * @param Nz Number of points in Z direction
     * @return Integrated value of |psi|^2
     */
    double integrateDeviceComplexNorm(double hx, double hy, double hz, const cuDoubleComplex *d_f, long Nx, long Ny, long Nz);

    /**
     * @brief Perform 3D integration of squared function using Simpson's rule (squares input on-the-fly)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
     * @param Nx Number of points in X direction (also used for indexing)
     * @param Ny Number of points in Y direction
     * @param Nz Number of points in Z direction
     * @return Integrated value of f^2
     */
    double integrateDeviceNorm(double hx, double hy, double hz, const double *d_f, long Nx, long Ny, long Nz);

    /**
     * @brief Perform 3D integration of psi^2 * coordinate^2 using Simpson's rule (computes on-the-fly)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
     * @param Nx Number of points in X direction (also used for indexing)
     * @param Ny Number of points in Y direction
     * @param Nz Number of points in Z direction
     * @param direction Direction for coordinate calculation (0=x, 1=y, 2=z)
     * @param scale Grid spacing for the chosen direction (dx, dy, or dz)
     * @return Integrated value of psi^2 * coordinate^2
     */
    double integrateDeviceRMS(double hx, double hy, double hz, const double *d_f, long Nx, long Ny, long Nz,
                              int direction, double scale);

    /**
     * @brief Perform 3D integration of all 3 RMS components in a single pass (3x more efficient)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f Pointer to function values (DEVICE memory, unpadded, uses Nx for indexing)
     * @param Nx Number of points in X direction (also used for indexing)
     * @param Ny Number of points in Y direction
     * @param Nz Number of points in Z direction
     * @param scale_x Grid spacing in X direction (dx)
     * @param scale_y Grid spacing in Y direction (dy)
     * @param scale_z Grid spacing in Z direction (dz)
     * @param[out] result_x2 Integrated value of psi^2 * x^2
     * @param[out] result_y2 Integrated value of psi^2 * y^2
     * @param[out] result_z2 Integrated value of psi^2 * z^2
     */
    void integrateDeviceRMSFused(double hx, double hy, double hz, const double *d_f, long Nx, long Ny, long Nz,
                                 double scale_x, double scale_y, double scale_z,
                                 double &result_x2, double &result_y2, double &result_z2);

    /**
     * @brief Perform 3D integration of all 3 RMS components in a single pass from complex array (3x more efficient)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f Pointer to function values (DEVICE memory, complex array, unpadded, uses Nx for indexing)
     * @param Nx Number of points in X direction (also used for indexing)
     * @param Ny Number of points in Y direction
     * @param Nz Number of points in Z direction
     * @param scale_x Grid spacing in X direction (dx)
     * @param scale_y Grid spacing in Y direction (dy)
     * @param scale_z Grid spacing in Z direction (dz)
     * @param[out] result_x2 Integrated value of |psi|^2 * x^2
     * @param[out] result_y2 Integrated value of |psi|^2 * y^2
     * @param[out] result_z2 Integrated value of |psi|^2 * z^2
     */
    void integrateDeviceComplexRMSFused(double hx, double hy, double hz, const cuDoubleComplex *d_f, long Nx, long Ny, long Nz,
                                        double scale_x, double scale_y, double scale_z,
                                        double &result_x2, double &result_y2, double &result_z2);

    /**
     * @brief Set the tile size for Z-direction processing
     * @param new_tile_size New number of z-slices per tile
     */
    void setTileSize(long new_tile_size);
};

#endif // SIMPSON3D_INTEGRATOR_HPP