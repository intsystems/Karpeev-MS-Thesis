# Jacobian Bound Refinements

This note summarizes the hierarchy of Jacobian bounds used in `exp4`, `exp5`,
and `exp6`.

## Setup

Let

```math
F(Y) = \operatorname{RMSNorm}(U(Y)),
```

where

```math
U(Y) = Y + \nu \bigl(C + TB(Y)\bigr).
```

The preactivation Jacobian is

```math
J_U(Y) = I + \nu J_{TB}(Y).
```

For a row-wise RMSNorm denominator, define

```math
\rho_i(U) = \operatorname{RMS}(U_{i,:})
          = \sqrt{\frac{1}{D}\sum_{j=1}^{D} U_{ij}^2 + \varepsilon}.
```

Let

```math
r_U = \min_i \rho_i(U).
```

## 1. Original Component Bound

The original theorem-style bound is

```math
\|J_F(Y)\|_2
\le
\frac{\gamma_{\max}}{r_U}
\left(
1
+ \nu
\frac{\gamma_{\max}^2}{r_{\mathrm{out}} r_{\mathrm{in}}}
\bigl(1+\|J_{\mathrm{FFN}}\|_2\bigr)
\bigl(1+\|J_{\mathrm{MSA}}\|_2\bigr)
\right).
```

This is a valid but conservative bound. It loses tightness mainly through

```math
\|I + J\|_2 \le 1 + \|J\|_2
```

and through multiplying separate worst-case component norms.

## 2. Residual-Component Bound

A tighter component-level refinement keeps the residual maps intact:

```math
\|J_F(Y)\|_2
\le
\frac{\gamma_{\max}}{r_U}
\left(
1
+ \nu
\frac{\gamma_{\max}^2}{r_{\mathrm{out}} r_{\mathrm{in}}}
\|I + J_{\mathrm{FFN}}\|_2
\|I + J_{\mathrm{MSA}}\|_2
\right).
```

This is always at least as tight as the original component bound because

```math
\|I + J_{\mathrm{FFN}}\|_2
\le
1 + \|J_{\mathrm{FFN}}\|_2,
```

and

```math
\|I + J_{\mathrm{MSA}}\|_2
\le
1 + \|J_{\mathrm{MSA}}\|_2.
```

## 3. TB-Norm Bound

If we measure the full Transformer sub-block Jacobian directly, we get

```math
\|J_F(Y)\|_2
\le
\frac{\gamma_{\max}}{r_U}
\left(
1 + \nu \|J_{TB}(Y)\|_2
\right).
```

This avoids the product of MSA and FFN worst-case directions.

## 4. Scalar Pre-RMSNorm Bound

The preactivation map is

```math
U(Y) = Y + \nu(C + TB(Y)).
```

Therefore

```math
J_U(Y) = I + \nu J_{TB}(Y).
```

Using only the final RMSNorm row-norm lower bound gives

```math
\|J_F(Y)\|_2
\le
\frac{\gamma_{\max}}{r_U}
\|J_U(Y)\|_2
=
\frac{\gamma_{\max}}{r_U}
\|I + \nu J_{TB}(Y)\|_2.
```

This is the `scalar pre-RMS bound` in `exp6`.

## 5. Row-Aware Outer RMSNorm Bound

The scalar pre-RMSNorm bound still uses only the worst row

```math
r_U = \min_i \rho_i(U).
```

A tighter row-aware bound keeps the row-wise denominators of the final RMSNorm.
Define

```math
D_U
=
\operatorname{blockdiag}_{i=1}^{S}
\left(
\frac{\gamma_{\max}}{\rho_i(U)} I_D
\right).
```

Then

```math
\boxed{
\|J_F(Y)\|_2
\le
\left\|
D_U
\left(I + \nu J_{TB}(Y)\right)
\right\|_2
}
```

This is the `row-aware pre-RMS bound` in `exp6`.

It is always no worse than the scalar pre-RMSNorm bound:

```math
\left\|
D_U
\left(I + \nu J_{TB}(Y)\right)
\right\|_2
\le
\|D_U\|_2
\left\|
I + \nu J_{TB}(Y)
\right\|_2
```

and

```math
\|D_U\|_2
=
\frac{\gamma_{\max}}{\min_i \rho_i(U)}
=
\frac{\gamma_{\max}}{r_U}.
```

Hence

```math
\left\|
D_U
\left(I + \nu J_{TB}(Y)\right)
\right\|_2
\le
\frac{\gamma_{\max}}{r_U}
\left\|
I + \nu J_{TB}(Y)
\right\|_2.
```

## 6. Exact Outer Factorization Check

For analysis only, one can freeze the exact Jacobian of the final RMSNorm at
`U`:

```math
J_{\mathrm{RMS}}(U)
=
\operatorname{blockdiag}_{i=1}^{S}
\left[
\operatorname{diag}(\gamma)
\left(
\frac{1}{\rho_i(U)} I_D
-
\frac{1}{D \rho_i(U)^3}
U_{i,:}^{\top} U_{i,:}
\right)
\right].
```

Then

```math
J_F(Y)
=
J_{\mathrm{RMS}}(U)
\left(I + \nu J_{TB}(Y)\right).
```

This is the `exact outer factorization` curve in `exp6`. It is a sanity check,
not an independent upper bound, because it is equal to the actual full-step
Jacobian factorization.

## Empirical Summary

On the pretrained DeiT-tiny linear-weight experiment at `nu = 2.0`:

```text
random context:
  empirical                 13.178
  scalar pre-RMS bound      13.602   gap 1.032x
  row-aware pre-RMS bound   13.190   gap 1.001x

image-derived context:
  empirical                 12.418
  scalar pre-RMS bound      12.800   gap 1.031x
  row-aware pre-RMS bound   12.432   gap 1.001x
```

The row-aware bound keeps the theorem-like RMSNorm damping structure while
removing almost all looseness caused by collapsing all rows to the minimum
normalization denominator.
