## Probability and Information Theory

This appendix provides the formal framework for probability and information theory using 
the language of topology, $\sigma$-algebras, and measure theory. We begin by defining events 
and event spaces as subsets of a sample space, introduce probability measures on these 
spaces, and then formalize the concept of information associated with events.

### Events

Given a set $\mathcal{C}$, an occurrence can be modeled as one of its subsets. The set of all possible occurrences is $\mathcal{P}(\mathcal{C})$. Not all occurrences may be of interest, so we may restrict ourselves to a subset.

An event space over $\mathcal{C}$ is a $\sigma$-algebra $\mathcal{A} \subseteq \mathcal{P}(\mathcal{C})$, whose elements are called events. This ensures:

- The trivial and impossible events are included:

$$
\emptyset \in \mathcal{A} \quad \text{and} \quad \mathcal{C} \in \mathcal{A}.
$$

- The non-occurrence of an event is also an event:

$$
A \in \mathcal{A} \Rightarrow A^C \in \mathcal{A}.
$$

- The combination of possible events is also an event:

$$
A_1, A_2, \dots \in \mathcal{A} \Rightarrow \bigcup_{\alpha} A_{\alpha} \in \mathcal{A}.
$$

If $(\mathcal{M}, \mathcal{T})$ is a topological space, we can construct an event space using the Borel $\sigma$-algebra:

$$
\mathcal{B} = \sigma(\mathcal{T}).
$$

### Probability

A probability function is a measure $P: \mathcal{A} \rightarrow [0,1]$ on an event space $\mathcal{A}$ that satisfies normalization:

$$
P(\mathcal{C}) = 1
$$

Then $(\mathcal{C}, \mathcal{A}, P)$ is called a probability space.

### Information

The information provided by an event $A \in \mathcal{A}$ can be defined based on:

- An event with probability 1 provides no information:

$$
P(A) = 1 \Rightarrow I(A) = 0.
$$

- Less probable events provide more information:

$$
P(A) < P(B) \Rightarrow I(A) > I(B).
$$

- For independent events observed separately, total information is additive:

$$
P(A \cap B) = P(A) P(B) \Rightarrow I(A \cap B) = I(A) + I(B).
$$

These rules lead to the standard definition:

$$
I(A) = -k \log P(A).
$$

where $k$ is a constant specifying the logarithm base.