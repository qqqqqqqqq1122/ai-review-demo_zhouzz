# Phase 1 Method Draft (ICLR-style): Dual-Channel Attribution for Dynamic GNNs

## 1. Problem Setup

Let a dynamic graph transition from baseline
\(G_0=(A_0, X_0)\) to target \(G_1=(A_1, X_1)\), where:

- \(A_t \in \mathbb{R}^{n\times n}\): weighted adjacency (or edge attributes),
- \(X_t \in \mathbb{R}^{n\times d}\): node feature matrix.

Define
\[
\Delta A = A_1 - A_0,\qquad \Delta X = X_1 - X_0.
\]

For a differentiable predictor \(f\), the target scalar output change is
\[
\Delta Y = f(A_1, X_1) - f(A_0, X_0).
\]

We treat entries of \((\Delta A, \Delta X)\) as players in a joint cooperative game.

## 2. Aumann-Shapley / Integrated Gradients on Joint Path

Use the linear interpolation path
\[
\gamma(\alpha) = \big(A(\alpha), X(\alpha)\big)
= \big(A_0 + \alpha \Delta A,\; X_0 + \alpha \Delta X\big),\quad \alpha\in[0,1].
\]

For any player coordinate \(i\) in the concatenated variable
\(z=(\mathrm{vec}(A),\mathrm{vec}(X))\), define joint attribution
\[
\phi_i^{\text{joint}}
= \int_0^1
\frac{\partial f(\gamma(\alpha))}{\partial z_i(\alpha)}
\,\frac{dz_i(\alpha)}{d\alpha}
\,d\alpha.
\]

Equivalent channel-wise decomposition:
\[
\Phi_A = \Delta A \odot \int_0^1 \nabla_A f\big(A(\alpha),X(\alpha)\big)d\alpha,
\quad
\Phi_X = \Delta X \odot \int_0^1 \nabla_X f\big(A(\alpha),X(\alpha)\big)d\alpha.
\]

Total attribution is
\[
\Phi_{\text{sum}} = \langle \mathbf{1}, \Phi_A \rangle + \langle \mathbf{1}, \Phi_X \rangle.
\]

## 3. Completeness (Summation-to-delta)

Assume \(f\) is differentiable almost everywhere along \(\gamma\) and integrable.
By chain rule:
\[
\frac{d}{d\alpha}f(\gamma(\alpha))
= \left\langle \nabla_A f(\gamma(\alpha)), \Delta A \right\rangle
+ \left\langle \nabla_X f(\gamma(\alpha)), \Delta X \right\rangle.
\]
Integrating from \(0\) to \(1\):
\[
\int_0^1 \frac{d}{d\alpha}f(\gamma(\alpha))d\alpha
= \int_0^1 \left\langle \nabla_A f(\gamma(\alpha)), \Delta A \right\rangle d\alpha
+ \int_0^1 \left\langle \nabla_X f(\gamma(\alpha)), \Delta X \right\rangle d\alpha.
\]
Left-hand side is
\[
f(\gamma(1)) - f(\gamma(0)) = \Delta Y.
\]
Right-hand side is exactly
\[
\langle \mathbf{1}, \Phi_A \rangle + \langle \mathbf{1}, \Phi_X \rangle.
\]
Hence,
\[
\sum_i \phi_i^{\text{joint}} = \Delta Y,
\]
which proves strict Summation-to-delta completeness.

## 4. Practical Monte Carlo / Riemann Approximation

With \(M\) steps (here \(M=50\)):
\[
\Phi_A \approx \Delta A \odot \frac{1}{M+1}\sum_{m=0}^{M}
\nabla_A f\big(A(\alpha_m), X(\alpha_m)\big),
\quad
\Phi_X \approx \Delta X \odot \frac{1}{M+1}\sum_{m=0}^{M}
\nabla_X f\big(A(\alpha_m), X(\alpha_m)\big),
\]
where \(\alpha_m = m/M\).

This is the implemented estimator for phase-two engineering validation.
