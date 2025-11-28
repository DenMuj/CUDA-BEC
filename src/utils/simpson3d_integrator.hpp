// simpson3d_integrator.hpp
#ifndef SIMPSON3D_INTEGRATOR_HPP
#define SIMPSON3D_INTEGRATOR_HPP

#include <cstddef>

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
     * @param d_f Pointer to function values (DEVICE memory)
     * @param Nx Number of points in X direction
     * @param Ny Number of points in Y direction
     * @param Nz Number of points in Z direction
     * @return Integrated value
     */
    double integrateDevice(double hx, double hy, double hz, double *d_f, long Nx, long Ny, long Nz);

    /**
     * @brief Perform 3D integration using Simpson's rule (complex array cast to double)
     * @param hx Step size in X direction
     * @param hy Step size in Y direction
     * @param hz Step size in Z direction
     * @param d_f Pointer to function values (DEVICE memory) - complex array cast to double
     * @param Nx Number of points in X direction
     * @param Ny Number of points in Y direction
     * @param Nz Number of points in Z direction
     * @return Integrated value
     */
    double integrateDeviceComplex(double hx, double hy, double hz, double *d_f, long Nx, long Ny, long Nz);

    /**
     * @brief Set the tile size for Z-direction processing
     * @param new_tile_size New number of z-slices per tile
     */
    void setTileSize(long new_tile_size);
};

#endif // SIMPSON3D_INTEGRATOR_HPP