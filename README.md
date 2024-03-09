# Selected states configuration interaction method (CIPSI)

# CIPSI alsgorithm

Configuration interaction (CI) method involves diagonalising the interacting Hamiltonian

$H=\sum_i \epsilon_i c_i^{\dagger}c_i+\frac{1}{2}\sum_{ijkl}V_{ijkl}c_i^{\dagger}c_j^{\dagger}c_kc_l$,

in the basis of all possible configurations of N electrons on M single particle (SP) states. The total dimension of the problem is given by

${M_u \choose N_u}{M_d \choose N_d}$,

where $M=M_u+M_d$ and $N=N_u+N_d$ for spins up (u) and down (d). Solving this problem with CI is only possible for small $N,M$.

CIPSI method involves diagonalising $H$ within only a subspace of the full configuration space, which includes only configurations relevant to the solution. 
The decision on whcih configurations are relevant is made by a perturbation-theory-based criterion.

CIPSI algorithm evaluates a contribution of configuration $q^{(0)}$ to an eigenstate $|p>\approx|p^{(0)}>$, which reads:

$\xi_{pq}=\frac{\langle q^{(0)}|H|q^{(0)} \rangle}{E_p^{(0)}-E_q^{(0)}}$.

When $\xi_{pq}$ is greater than a given threshold, configuration $q^{(0)}$ is added to the subspace. This continues iteratively until no configurations are added.

## Input files

The input files for a problem of interacting electrons in MoS2 quantum dot consist of
* SP energy levels for spin up (U) and down (D)
* CME elements for spin combinations UU, DD, UD for bare Coulomb interaction and Keldysh screening
* quantum numbers specific to the problem of MoS2 QD: valley index and angular momentum



## References

[1] M. Bieniek, L. Szulakowska, and P. Hawrylak, “Effect of valley, spin, and band nesting on the electronic properties of gated quantum dots in a single layer of transition metal dichalcogenides,” Phys. Rev. B, vol. 101, no. 3, p. 035401, Jan. 2020, doi: 10.1103/PhysRevB.101.035401.

[2] L. Szulakowska, M. Cygorek, M. Bieniek, and P. Hawrylak, “Valley- and spin-polarized broken-symmetry states of interacting electrons in gated Mo S 2 quantum dots,” Phys. Rev. B, vol. 102, no. 24, p. 245410, Dec. 2020, doi: 10.1103/PhysRevB.102.245410.

[3] M. Cygorek, “Accurate and efficient description of interacting carriers in quantum nanostructures by selected configuration interaction and perturbation theory,” PHYSICAL REVIEW B, p. 10, 2020.


