# MarginFlow

**Estimate latent transitions from grouped marginal data.**

MarginFlow estimates how a population may have moved between categorical
states when only group totals from two points in time are available. Examples
include movement between customer tiers, employment categories, health states,
survey responses, or electoral choices.

## The idea

For each group, $X$ contains source-category counts and $Y$ contains
target-category counts. MarginFlow estimates a shared transition matrix $P$,
where each entry is the probability of moving from one source category to one
target category:

$$
P_{ij}=\Pr(\text{target}=j\mid\text{source}=i).
$$

Every probability is non-negative and every row sums to one. The model predicts
target proportions with $Y\approx XP$ and fits $P$ by constrained maximum
likelihood. Estimated aggregate flows are the source totals multiplied by the
corresponding transition probabilities.

Marginal totals do not uniquely reveal individual transitions. The result is a
model-based estimate, not a reconstruction of individual records. It is most
informative when groups have varied source compositions and represent the same
population at both times.

## Proof on synthetic data

One deterministic unit test uses three groups whose result can be checked by
inspection:

$$
X=
\begin{bmatrix}
1000 & 0 \\
0 & 1000 \\
500 & 500
\end{bmatrix},
\qquad
Y=
\begin{bmatrix}
800 & 200 \\
300 & 700 \\
550 & 450
\end{bmatrix}.
$$

The first group contains only source A, so its 800:200 target split directly
reveals the first transition row, $(0.8,0.2)$. The second contains only source
B and reveals $(0.3,0.7)$. The third is a 50:50 mix, so its expected target
counts are their average:

$$
\left(\frac{800+300}{2},\frac{200+700}{2}\right)=(550,450).
$$

The known matrix is therefore:

$$
P=
\begin{bmatrix}
0.8 & 0.2 \\
0.3 & 0.7
\end{bmatrix}.
$$

Using only $X$ and $Y$, MarginFlow infers:

$$
\widehat{P}=
\begin{bmatrix}
0.800000 & 0.200000 \\
0.300000 & 0.700000
\end{bmatrix},
\qquad
\operatorname{MAE}(\widehat{P},P)<10^{-8}.
$$

## Use the library

```python
import numpy as np

from marginflow import MarginFlowModel

X = np.array([[1000, 0], [0, 1000], [500, 500]])
Y = np.array([[800, 200], [300, 700], [550, 450]])

model = MarginFlowModel().fit(X, Y)
print(model.transition_matrix_)
```

Synthetic recovery verifies the implementation when its model assumptions are
true; it does not prove those assumptions for real populations.
