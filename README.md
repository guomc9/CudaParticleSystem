# CudaParticleSystem

## Force Field
### Interface
...
### Implemetation Examples
#### Whirlwind Field
##### Radial force
$$
F_r = -\frac{\mathrm{d}P}{\mathrm{d}r}\cdot A
$$
##### Axial force
$$
F_z = ma_z
$$
##### Tangential force
$$
F_\tau = m\cdot \omega^2 \cdot r
$$
##### Parameters
$m$ : Mass of a particle.

$r$ : Radial distance from whirlwind center to particle.

$A$ : Force area of particle, assume that $A=\pi\cdot R^2$, $R$ is the radius of a particle.

$\frac{\mathrm{d}P}{\mathrm{d}r}$ : Pressure gradient, assume that $\frac{\mathrm{d}P}{\mathrm{d}r}=-2\alpha rP_0e^{-2\alpha r^2}$, $\alpha$ and $P_0$ is a constant.

$a_z$ : Axial acceleration, assume that $a_z=\beta\cdot e^{-\lambda |z|}$, $\beta$ is a constant.

$\omega$ : Angular velocity, assume that $\omega$ is a constant.

#### Gravity Field
##### Axial force
$$
F_z = -mg


