# Solver-in-the-Loop

This is the source code repository for our paper
["Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers"](https://github.com/tum-pbs/Solver-in-the-Loop),
stay tuned...

![3D Unsteady Wake Flow: Source vs Learned Correction vs Reference](resources/SoL-karman-3d-sideBySide.gif)

## Abstract:

Finding accurate solutions to partial differential equations (PDEs) is a crucial task in all scientific and engineering disciplines. It has recently been shown that machine learning methods can improve the solution accuracy by correcting for effects not captured by the discretized PDE. We target the problem of reducing numerical errors of iterative PDE solvers and compare different learning approaches for finding complex correction functions. We find that previously used learning approaches are significantly outperformed by methods that integrate the solver into the training loop and thereby allow the model to interact with the PDE during training. This provides the model with realistic input distributions that take previous corrections into account, yielding improvements in accuracy with stable rollouts of several hundred recurrent evaluation steps and surpassing even tailored supervised variants. We highlight the performance of the differentiable physics networks for a wide variety of PDEs, from non-linear advection-diffusion systems to three-dimensional Navier-Stokes flows.

<https://ge.in.tum.de/>,
<https://perso.telecom-paristech.fr/kum/>

# Tutorial

**Requirements**

- [TensorFlow](https://www.tensorflow.org/); *tested with 1.15*
- [PhiFlow](https://github.com/tum-pbs/PhiFlow); *tested with commit-4f5e678*

**Running tests**

Please find Makefile in each folder. A set of targets is provided for each scenario.
For example, in karman-2d, you can generate data sets, train a model, and apply as follows:
```
make karman-fdt-hires-set      # Genrate traning data set
make karman-fdt-hires-testset  # Genrate test data set
make karman-fdt-sol32          # Train a model
make karman-fdt-sol32/run_test # Run test
```
