# brane-count

Methods to count the number of intersecting D6-brane vacua of type IIA on the $\mathbb{T}^6/\mathbb{Z}_2\times\mathbb{Z}_2$ orientifold. Vacua consist of stacks of $N_a$ coincident D6-branes wrapping factorized 3-cycles described by three pairs of coprime winding numbers, $(n_a^i,m_a^i)$. The homology classes can be written as $[\Pi_a] = \sum_I(\widehat{X}_a^I[\pi_I^+] + \widehat{Y}_a^I[\pi_I^-])$, where $\widehat{X}^0=n^1n^2n^3$, $\widehat{X}^1=-n^1\widehat{m}^2\widehat{m}^3,\ldots$ with $\widehat{m}^i=m^i+2b_i(n^i+m^i)\in\mathbb{Z}$. These topological data are subject to several consistency conditions which can be written as follows:

- Tadpole cancellation: $\sum_aN_a\widehat{X}_a^I = 8$ for each $I=0,1,2,3$
- K-theory charge cancellation: $\sum_aN_a\widehat{Y}_a^I \in 2\mathbb{Z}$ for each $I=0,1,2,3$
- Supersymmetry: $\sum_I\widehat{X}_a^I\widehat{U}_I > 0$ and $\sum_I\frac{\widehat{Y}_a^I}{\widehat{U}_I} = 0$ for all $a$

The discrete NSNS fluxes take values $b_{1,2,3}\in\\{0,\frac{1}{2}\\}$ and the moduli $\widehat{U}_I$ must be positive.

That there are only a finite number of solutions to the above system of Diophantine equations has been known since the work of [Douglas and Taylor ('06)](https://doi.org/10.1088/1126-6708/2007/01/031). In

> Gregory J. Loges and Gary Shiu, _134 billion intersecting brane models_, J. High Energy Phys. **2022**, 97 (2022).
> doi:[10.1007/JHEP12(2022)097](https://doi.org/10.1007/JHEP12(2022)097) arXiv:[2206.03506](https://doi.org/10.48550/arXiv.2206.03506)

we develop techniques to provide an _exact_ count of gauge-inequivalent vacua.
