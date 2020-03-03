/* C++ headers */
#include <algorithm>  // min, max
#include <assert.h>   // Assertions
#include <ctime>      // Computer clock (used for prng)

/* Athena++ headers */
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../defs.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"

/** Lower Y boundary condition. The 'ghost zone' is an offscreen line
 *  of cells that produces hot gas. */
void LowerYGhostZone(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                    Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

/** The cooling function that models radiation in the interstellar medium. */
void Cooling(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                          const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

/** Define variables that need to be used in multiple functions. */
namespace {
Real tHot, tWarm, nHot, nWarm; // Temperature and number density
Real vHot, rhoHot, rhoWarm; // Velocity and mass density
Real pHot, pWarm, gm1; // Pressure, gamma constant
Real gasRatio; //The y position of the gas interface as a percentage of box height
int numModes, minMode, modeGap; // Information about random perturbation
}

/** Constants used throughout the generator. */
static const Real kB = 1.3806485279e-16;      // Boltzmann's constant (erg/K)
static const Real mBar = 1.27 * 1.6733e-24;   // Average mass density
static const Real pi = 3.1415926535897932384; // Pi

/** Inform Athena++ about the custom boundary condition 
 *  and cooling function. */
void Mesh::InitUserMeshData(ParameterInput *pin) {
    EnrollUserExplicitSourceFunction(Cooling);
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, LowerYGhostZone);
    return;
}

/** Flag the number of user outputs and 
 *  assign index numbers and names to each. */
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
    AllocateUserOutputVariables(6);
    SetUserOutputVariableName(0, "Temperature");
    SetUserOutputVariableName(1, "Mass Density");
    SetUserOutputVariableName(2, "Momentum X");
    SetUserOutputVariableName(3, "Momentum Y");
    SetUserOutputVariableName(4, "Momentum Z");
    SetUserOutputVariableName(5, "Pressure");
    return;
}

/** Calculate temperature and output it. We can only output
 * 'user variables', so we assign other key values as 
 * user output variables. */
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
    for(int k = ks; k <= ke; k++) {
        for(int j = js; j <= je; j++) {
            for(int i = is; i <= ie; i++) {
                user_out_var(0, k, j, i) = mBar * phydro->w(IPR, k, j, i) / 
                                           (kB * phydro->w(IDN, k, j, i)); // Temperature
                user_out_var(1, k, j, i) = phydro->w(IDN, k, j, i); // Density
                user_out_var(2, k, j, i) = phydro->w(IM1, k, j, i); // Momenta
                user_out_var(3, k, j, i) = phydro->w(IM2, k, j, i);
                user_out_var(4, k, j, i) = phydro->w(IM3, k, j, i);
                user_out_var(5, k, j, i) = phydro->w(IPR, k, j, i); // Pressure
            }
        }
    }
}

/** Define the body of our boundary condition. If we are below the bottom
 *  of the screen (y-axis) then set initial conditions to match the hot gas. */
void LowerYGhostZone(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                    Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
    for (int k=kl; k<=ku; ++k) {
        for (int j=1; j<=ngh; ++j) {
            for (int i=il; i<iu; ++i) {
                prim(IDN,k,jl-j,i) = rhoHot;
                prim(IVX,k,jl-j,i) = vHot;
                prim(IVY,k,jl-j,i) = 0.0;
                prim(IVZ,k,jl-j,i) = 0.0;
                prim(IPR,k,jl-j,i) = pHot;
            }
        }
    }
}

/** Define the body of our cooling function. Find our change in temperature
 *  dTemp according to known results, approriately subract from energy. */
void Cooling(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                          const AthenaArray<Real> &bcc, AthenaArray<Real> &cons) {
    Real dTemp = 0;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
            for (int i=pmb->is; i<=pmb->ie; ++i) {
                Real t = mBar * prim(IPR,k,j,i) / (kB * prim(IDN,k,j,i));
                if (t <= 1e4) {
                    dTemp = 0;
                } else if (t < 1e5) {
                    dTemp = 3.1620e-30 * std::pow(t, 1.6);
                } else if (t < 2.884e5) {
                    dTemp = 3.1620e-21 * std::pow(t, -0.2);
                } else if (t < 4.732e5) {
                    dTemp = 6.31e-6 * std::pow(t, -3);
                } else if (t < 2.113e6) {
                    dTemp = 1.047e-21 * std::pow(t, -0.22);
                } else if (t < 3.981e6) {
                    dTemp = 3.981e-4 * std::pow(t, -3);
                } else if (t < 1.995e7) {
                    dTemp = 4.169e-26 * std::pow(t, 0.33);
                } else {
                    dTemp = 2.399e-27 * std::pow(t, 0.5);
                }
                cons(IEN,k,j,i) -= dt * dTemp * std::pow(prim(IDN,k,j,i)/mBar, 2);
            }
        }
    }
    return;
}

/** Initialize the simulation. Collect values from the input file,
 *  then initialize according to Esquivel et al. */
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    // Values from the input file
    gm1 = peos->GetGamma() - 1.0;
    tHot = pin->GetReal("problem","T_hot");
    tWarm = pin->GetReal("problem","T_warm");
    nHot = pin->GetReal("problem","n_hot");
    nWarm = pin->GetReal("problem","n_warm");
    vHot = pin->GetReal("problem","v_hot");
    gasRatio = pin->GetReal("problem","gas_ratio");
    numModes = pin->GetInteger("problem","num_modes");
    minMode = pin->GetInteger("problem","min_mode");
    modeGap = pin->GetInteger("problem","mode_gap");
    Real boxWidth = (pin->GetReal("mesh","x1max") - pin->GetReal("mesh","x1min"));
    Real boxHeight = (pin->GetReal("mesh","x2max") - pin->GetReal("mesh","x2min"));
    Real mag = pin->GetReal("problem","mag");

    // Quantities derived from the input file
    rhoHot = nHot * mBar;
    rhoWarm = nWarm * mBar;
    pHot = nHot * kB * tHot;
    pWarm = nWarm * kB * tWarm;
    Real a = mag * boxWidth;
    Real dV = vHot / 100;
    Real interPos = pin->GetReal("mesh","x2max") - (boxHeight * gasRatio);
    

    // Arrays needed to store our normals for perturbation
    Real waveLengths[numModes];
    Real phases[numModes];

    // Seed the prng with the current time
    std::srand(std::time(nullptr));

    // Assign wave lengths based on the input file, phases randomly
    for (int i = 0; i < numModes; i++) {
        waveLengths[i] = boxWidth / (minMode + i * modeGap);
        phases[i] = std::rand() / RAND_MAX * boxWidth / 2;
    }

    // Check for pressure equilibrium, otherwise throw an error
    assert(tHot * nHot == tWarm * nWarm);

    for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
            Real interDist = pcoord->x2v(j) - interPos;
            for (int i=is; i<=ie; i++) {
                // Setting mass density with a hyperbolic tangent transition
                phydro->u(IDN,k,j,i) = rhoHot + (rhoWarm - rhoHot) / 2.0 * (1 + std::tanh(interDist / a));
                // Setting x momentum with a hyperbolic tangent transition
                phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i) * vHot / 2.0 * (1 - std::tanh(interDist / a));
                // Build the sines based on the wavelengths and phases
                Real sines;
                for (int s = 0; s < numModes; s++) {
                    sines += std::sin(2 * pi * (pcoord->x1v(i) / waveLengths[s] + phases[s]));
                }
                sines /= numModes;
                // Set the initial perturbation in the y direction
                phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i) * dV * std::exp(-1 * std::pow((interDist / a), 2)) * sines;
                // No initial velocity in the z direction
                phydro->u(IM3,k,j,i) = 0.0;
                // Set initial thermal and kinetic energy
                phydro->u(IEN,k,j,i) = pHot/gm1 + 0.5 * (SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i))) / phydro->u(IDN,k,j,i);
            }
        }
    }
}