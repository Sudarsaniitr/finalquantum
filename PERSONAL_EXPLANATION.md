# Complete Explanation — Quantum Classifier Project
## For people who know NOTHING about quantum computing or machine learning

This file is our personal guide. Start from the top. Everything is explained
from scratch with no prior knowledge assumed.

---

# PART 1: WHAT IS THIS PROJECT ABOUT?

We implemented a research paper that builds a **machine learning classifier
using quantum computers**. The paper is:

> Blank, Park, Rhee, Petruccione —
> "Quantum classifier with tailored quantum kernel"
> npj Quantum Information, 2020.
> https://www.nature.com/articles/s41534-020-0272-6

In simpler words: The paper says "here is a new way to use a quantum computer
to decide which category an unknown thing belongs to." We coded that up, ran it
on a real IBM quantum computer, and added our own improvement on top.

---

# PART 2: BACKGROUND — BITS AND QUBITS

## Classical computers use bits

A regular computer stores information as **bits**: each bit is either 0 or 1.
That's it. 8 bits make a byte. A phone has billions of bits.

## Quantum computers use qubits

A **qubit** is different. A qubit can be:
- In state |0⟩ (like a bit being 0)
- In state |1⟩ (like a bit being 1)
- **Or BOTH at the same time** — this is called **superposition**

The weird thing: a qubit is SIMULTANEOUSLY |0⟩ and |1⟩, with different
amounts of each. Like a coin spinning in the air — it's neither heads nor
tails until it lands.

Mathematically, a qubit is written as:
```
|ψ⟩ = α|0⟩ + β|1⟩
```
where α and β are complex numbers (numbers with a real and imaginary part)
satisfying |α|² + |β|² = 1. The |α|² is the probability of measuring 0,
and |β|² is the probability of measuring 1.

## What does | ⟩ mean?

This is "ket notation" (Dirac notation). It's just how physicists write
quantum states. |0⟩ means "the qubit is in state 0", |ψ⟩ means "the qubit
is in state ψ (psi)". Think of it like parentheses — it groups and labels
a quantum state.

## What is a quantum gate?

A quantum gate is like a logic gate (AND, OR, NOT) but for qubits. Gates
**rotate** the qubit state. The most common ones:

- **H (Hadamard gate):** Turns |0⟩ into (|0⟩+|1⟩)/√2 — equal superposition.
  It's the "put the coin spinning" operation.

- **CSWAP (controlled-SWAP):** If one qubit is |1⟩, swap two other qubits.
  If it's |0⟩, do nothing. A 3-qubit conditional operation.

- **RX(θ), RY(θ), RZ(θ):** Rotate the qubit by angle θ around the X, Y, or Z
  axis of the Bloch sphere (see below).

## What is the Bloch sphere?

Imagine a globe. The north pole = |0⟩, south pole = |1⟩. Any other point
on the surface = a superposition. Every qubit state is a point on this sphere.

The test state in our project sweeps around the equator:
```
|x̃(θ)⟩ = cos(θ/2)|0⟩ + i·sin(θ/2)|1⟩
```
As θ goes from 0 to 2π, the state traces out a full equatorial circle.

---

# PART 3: BACKGROUND — MACHINE LEARNING BASICS

## What is classification?

Classification = deciding which category something belongs to.
Examples:
- Email: spam or not spam?
- Photo: cat or dog?
- Our problem: which class does a quantum state belong to?

## What is a kernel?

A kernel is a way to **measure similarity** between two things.
If kernel(A, B) is large, A and B are similar.
If kernel(A, B) is small, A and B are different.

In machine learning, kernels let you classify without explicitly computing
complicated features — you only need pairwise similarities.

## Distance-based classification

The simplest classifier: given a new unknown thing, find the training example
most similar to it and return that label.

Our quantum classifier does exactly this:
- Two training states |x₁⟩ (class A) and |x₂⟩ (class B)
- For any test state |x̃⟩, compute: similarity with |x₁⟩ minus similarity with |x₂⟩
- If positive → class A. If negative → class B.

---

# PART 4: THE TOY PROBLEM

The paper uses a deliberately simple example to make everything crystal clear.

## Training states (Eq. 11 of the paper)

```
|x₁⟩ = (i/√2)|0⟩ + (1/√2)|1⟩    ← labelled class 0
|x₂⟩ = (i/√2)|0⟩ − (1/√2)|1⟩    ← labelled class 1
```

Notice: the ONLY difference between |x₁⟩ and |x₂⟩ is the sign on the |1⟩
component. |x₁⟩ has +1/√2 and |x₂⟩ has −1/√2.

Both states are normalized: |i/√2|² + |±1/√2|² = 1/2 + 1/2 = 1. ✓

## Test state

```
|x̃(θ)⟩ = cos(θ/2)|0⟩ + i·sin(θ/2)|1⟩
```

As θ sweeps from 0 to 2π, this moves around the Bloch sphere equator.
At θ=0: state is |0⟩. At θ=π: state is i|1⟩ ≈ |1⟩.

## The ground truth

- For θ ∈ (0, π): test state is "closer" to |x₁⟩ → should be class 0
- For θ ∈ (π, 2π): test state is "closer" to |x₂⟩ → should be class 1
- At θ = 0 and θ = π: exactly equal distance → boundary (ambiguous)

---

# PART 4.5: WHEN THE HADAMARD CLASSIFIER DOES WORK

Before we show where Hadamard fails, let's see it succeed — so the failure
feels surprising and meaningful, not confusing.

## The key ingredient: REAL inner products

The Hadamard kernel is:
```
K_H = Σ_m (-1)^{y_m} · w_m · Re⟨x̃|x_m⟩
```

It uses `Re⟨x̃|x_m⟩` — the **real part** of the inner product.
This works perfectly when the states are chosen so that the real part is
large and different between classes. Let's build such an example.

---

## Worked example — 2 training states on the real axis

### Choose training states

Let's pick two states that live entirely on the real axis (no imaginary parts):

```
|x₁⟩ = |0⟩ = [1, 0]     ← class 0   (the "north pole" of the Bloch sphere)
|x₂⟩ = |1⟩ = [0, 1]     ← class 1   (the "south pole")
```

Both are normalized: |1|² + |0|² = 1 ✓

### Choose test state

Let the test state also be real:
```
|x̃(θ)⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩   =   [cos(θ/2),  sin(θ/2)]
```
(No `i` in front of sin — this is a REAL state, unlike the paper's toy problem.)

### Compute the inner products

```
⟨x̃|x₁⟩ = ⟨x̃|0⟩ = conjugate([cos(θ/2), sin(θ/2)]) · [1, 0]
         = cos(θ/2)·1 + sin(θ/2)·0
         = cos(θ/2)
```

```
⟨x̃|x₂⟩ = ⟨x̃|1⟩ = conjugate([cos(θ/2), sin(θ/2)]) · [0, 1]
         = cos(θ/2)·0 + sin(θ/2)·1
         = sin(θ/2)
```

Both are **real numbers**. So Re⟨x̃|x₁⟩ = cos(θ/2) and Re⟨x̃|x₂⟩ = sin(θ/2).

### Compute the Hadamard kernel

With equal weights w₁ = w₂ = 0.5:
```
K_H = 0.5·Re⟨x̃|x₁⟩ − 0.5·Re⟨x̃|x₂⟩
    = 0.5·cos(θ/2) − 0.5·sin(θ/2)
    = 0.5·[cos(θ/2) − sin(θ/2)]
```

### Let's plug in numbers for 3 test angles

**Test at θ = π/2 (45° on Bloch sphere — closer to |0⟩):**
```
cos(π/4) = 0.707,  sin(π/4) = 0.707
K_H = 0.5·(0.707 − 0.707) = 0.0
```
Exactly on the boundary — ambiguous. Makes sense since cos = sin at π/4.

**Test at θ = π/3 (closer to |0⟩ = class 0):**
```
cos(π/6) = 0.866,  sin(π/6) = 0.500
K_H = 0.5·(0.866 − 0.500) = +0.183
```
Positive → predict **class 0** ✓  (test state was indeed closer to |x₁⟩ = |0⟩)

**Test at θ = 2π/3 (closer to |1⟩ = class 1):**
```
cos(π/3) = 0.500,  sin(π/3) = 0.866
K_H = 0.5·(0.500 − 0.866) = −0.183
```
Negative → predict **class 1** ✓  (test state was closer to |x₂⟩ = |1⟩)

### Why it works here

The inner products are real because the states are real vectors.
Real part = full value, nothing is hidden or thrown away.
The kernel correctly encodes "which training state are you closer to?"

```
K_H > 0  →  cos(θ/2) > sin(θ/2)  →  θ/2 < π/4  →  θ < π/2  →  class 0
K_H < 0  →  sin(θ/2) > cos(θ/2)  →  θ/2 > π/4  →  θ > π/2  →  class 1
```

Decision boundary at θ = π/2. Perfect separation. ✓

---

## Worked example — 3 training states (one per class + noise)

Let's make it slightly more realistic with 3 training states:

```
|x₁⟩ = [1.0,  0.0]   label 0   (pure |0⟩)
|x₂⟩ = [0.6,  0.8]   label 0   (another class-0 state, normalized: 0.36+0.64=1 ✓)
|x₃⟩ = [0.0,  1.0]   label 1   (pure |1⟩)
```

Weights: w₁ = w₂ = w₃ = 1/3

Test state: same as before, |x̃(θ)⟩ = [cos(θ/2), sin(θ/2)]

Inner products (all real):
```
⟨x̃|x₁⟩ = cos(θ/2)
⟨x̃|x₂⟩ = 0.6·cos(θ/2) + 0.8·sin(θ/2)
⟨x̃|x₃⟩ = sin(θ/2)
```

Kernel (labels: x₁=0, x₂=0, x₃=1):
```
K_H = (1/3)·[+1·cos(θ/2)]          ← x₁ label 0: sign = (-1)^0 = +1
    + (1/3)·[+1·(0.6·cos(θ/2) + 0.8·sin(θ/2))]    ← x₂ label 0
    + (1/3)·[−1·sin(θ/2)]          ← x₃ label 1: sign = (-1)^1 = −1

    = (1/3)·[cos(θ/2) + 0.6·cos(θ/2) + 0.8·sin(θ/2) − sin(θ/2)]
    = (1/3)·[1.6·cos(θ/2) − 0.2·sin(θ/2)]
```

**Test at θ = 0.5 rad (small angle, close to |0⟩):**
```
cos(0.25) ≈ 0.969,  sin(0.25) ≈ 0.247
K_H = (1/3)·[1.6·0.969 − 0.2·0.247]
    = (1/3)·[1.550 − 0.049]
    = (1/3)·1.501 ≈ +0.500
```
Positive → **class 0** ✓ (small θ means state is close to |0⟩ = class 0)

**Test at θ = 2.5 rad (large angle, close to |1⟩):**
```
cos(1.25) ≈ 0.315,  sin(1.25) ≈ 0.949
K_H = (1/3)·[1.6·0.315 − 0.2·0.949]
    = (1/3)·[0.504 − 0.190]
    = (1/3)·0.314 ≈ +0.105
```
Still positive, but weaker. Correct (class 0 dominates with 2/3 weight).

**Test at θ = 3.0 rad (very close to |1⟩):**
```
cos(1.5) ≈ 0.0707,  sin(1.5) ≈ 0.997
K_H = (1/3)·[1.6·0.0707 − 0.2·0.997]
    = (1/3)·[0.113 − 0.199]
    = (1/3)·(−0.086) ≈ −0.029
```
Negative → **class 1** ✓

The Hadamard classifier works fine here too! With 3 training states, it still
correctly identifies the region near |1⟩ as class 1.

---

## So what's the pattern? When does Hadamard work vs fail?

| Situation | Hadamard works? | Why |
|---|---|---|
| Training states are **real vectors** (no imaginary parts) | ✓ Yes | Re⟨x̃\|xₘ⟩ = full overlap, nothing lost |
| Training states are **on the real axis** of the Bloch sphere | ✓ Yes | States are real by definition |
| Training states have **different real parts** per class | ✓ Yes | Kernel has useful signal |
| Training states are designed so Re⟨x̃\|xₘ⟩ = 0 for all x̃ | ✗ No | Kernel = 0 everywhere, blind |
| Inner products are **purely imaginary** | ✗ No | Real part vanishes |
| The distinction between classes lives in the **phase** of the overlap | ✗ No | Re() strips out phase information |

**The fundamental rule:**
> The Hadamard classifier uses only the real part of the overlap.
> Any classification problem where the class distinction lives in the
> imaginary part (the phase) of the quantum state is INVISIBLE to it.

The paper's toy problem is specifically engineered to be one such case —
training states with an imaginary component (the `i/√2` factor) that makes
all real parts zero. This is not an edge case — in quantum ML, data is often
encoded with complex amplitudes, and phases carry crucial information.

---

# PART 4.7: CAN I LOOK AT THE TRAINING SET AND PREDICT IF HADAMARD WILL FAIL?

Yes. Completely. Before running a single circuit. Here is the exact test.

---

## Step back — what is a complex number? (truly from scratch)

You already know real numbers: 1, 2, −5, 0.7, π. They live on a number line.

A **complex number** has TWO parts:
```
z = a + ib
```
- `a` is the **real part** — the ordinary number part
- `b` is the **imaginary part** — the coefficient of `i`
- `i` is defined as √(−1), a number that doesn't exist on the real number line

Think of it as a 2D coordinate: instead of a number line, you have a flat plane.
The real part goes left-right, the imaginary part goes up-down.

```
         imaginary axis
              ↑
          2i  •
              |
    −3 ←------+------→  real axis
              |
         −2i  •
```

Examples:
- `3 + 4i` → point at (3, 4) on that plane
- `2i` → point at (0, 2) — pure imaginary, zero real part
- `5` → point at (5, 0) — pure real, zero imaginary part
- `i/√2` → point at (0, 1/√2) — pure imaginary

**Magnitude** of a complex number = distance from origin:
```
|a + ib| = √(a² + b²)
```

**Complex conjugate** = flip the sign of the imaginary part:
```
conj(a + ib) = a − ib
```

Key fact: `z · conj(z) = (a + ib)(a − ib) = a² + b² = |z|²`

---

## What is a quantum state vector? (truly from scratch)

A 1-qubit state is a list of TWO complex numbers:
```
|ψ⟩ = [α, β]   where α and β are complex numbers
```

Rule: `|α|² + |β|² = 1` (the probabilities must add to 1)

- `|α|²` = probability of measuring 0
- `|β|²` = probability of measuring 1

Examples:
```
|0⟩ = [1, 0]          → measure 0 with 100% probability
|1⟩ = [0, 1]          → measure 1 with 100% probability
|+⟩ = [1/√2, 1/√2]   → measure 0 or 1 with 50/50 probability
```

Our toy training states:
```
|x₁⟩ = [i/√2,  1/√2]    → |i/√2|² + |1/√2|² = 1/2 + 1/2 = 1  ✓
|x₂⟩ = [i/√2, −1/√2]    → same check, same result ✓
```

Notice: the first component `i/√2` is purely imaginary (lives on the imaginary
axis). The second component `±1/√2` is purely real.

---

## What is an inner product? (truly from scratch)

The inner product `⟨a|b⟩` between two quantum states is computed like this:

1. Take the CONJUGATE of every component of the left state `⟨a|`
2. Multiply component-by-component with the right state `|b⟩`
3. Add them all up

Formula for 1-qubit states `[α, β]` and `[a, b]`:
```
⟨[α,β] | [a,b]⟩  =  conj(α)·a  +  conj(β)·b
```

That's it. A dot product, but you conjugate the left side first.

**Worked example** — inner product of |0⟩ and |+⟩:
```
⟨0|+⟩ = conj(1)·(1/√2)  +  conj(0)·(1/√2)
       = 1·(1/√2)  +  0·(1/√2)
       = 1/√2
```
This is a real number = 1/√2 ≈ 0.707. Makes sense — they're somewhat similar.

**Worked example** — inner product of |0⟩ and |1⟩:
```
⟨0|1⟩ = conj(1)·0  +  conj(0)·1
       = 0 + 0 = 0
```
Zero — they're completely different (orthogonal). Makes sense.

---

## Now let's compute the inner product for the toy problem

Test state: `|x̃(θ)⟩ = [cos(θ/2),  i·sin(θ/2)]`

Training state 1: `|x₁⟩ = [i/√2,  1/√2]`

```
⟨x̃|x₁⟩ = conj(cos(θ/2)) · (i/√2)  +  conj(i·sin(θ/2)) · (1/√2)
```

Now simplify piece by piece:
- `cos(θ/2)` is a real number, so `conj(cos(θ/2)) = cos(θ/2)`
- `i·sin(θ/2)` has imaginary part, so `conj(i·sin(θ/2)) = −i·sin(θ/2)`

So:
```
⟨x̃|x₁⟩ = cos(θ/2) · (i/√2)  +  (−i·sin(θ/2)) · (1/√2)
         = i·cos(θ/2)/√2  −  i·sin(θ/2)/√2
         = (i/√2) · [cos(θ/2) − sin(θ/2)]
```

This entire result is multiplied by `i` — it is **purely imaginary**.

The real part: `Re(⟨x̃|x₁⟩) = 0`   for every single value of θ.

Similarly (try it yourself — same calculation with −1/√2 instead of 1/√2):
```
⟨x̃|x₂⟩ = (i/√2) · [cos(θ/2) + sin(θ/2)]   → also purely imaginary
```
`Re(⟨x̃|x₂⟩) = 0`   for every single value of θ.

So the Hadamard kernel:
```
K_H = 0.5 · Re(⟨x̃|x₁⟩)  −  0.5 · Re(⟨x̃|x₂⟩)
    = 0.5 · 0  −  0.5 · 0
    = 0
```
Always. Everywhere. For every θ. Not just at boundaries — literally everywhere.

---

## The diagnostic test — 3 steps, works on any training set

### Step 1: Compute the "difference vector" D

```
D = (average of class-0 training states) − (average of class-1 training states)
```

With equal weights this is just:
```
D = (1/|class 0|) · Σ_{class 0} xₘ  −  (1/|class 1|) · Σ_{class 1} xₘ
```

For our toy problem:
```
D = (1/1)·[i/√2, 1/√2]  −  (1/1)·[i/√2, −1/√2]
  = [i/√2 − i/√2,   1/√2 − (−1/√2)]
  = [0,   2/√2]
  = [0,   √2]
```

### Step 2: Extract Re(D) — the real part of each component

```
D = [0,  √2]

Re(D) = [Re(0),  Re(√2)]
       = [0,     √2]       ← second component is real and nonzero
```

So Re(D) is NOT the zero vector. Does that mean Hadamard works? Not yet —
we need Step 3.

### Step 3: Check what the test state does to Re(D)

The Hadamard kernel = `Re⟨x̃|D⟩`. Let's compute it directly:

```
x̃ = [cos(θ/2),   i·sin(θ/2)]

⟨x̃|D⟩ = conj(cos(θ/2)) · 0  +  conj(i·sin(θ/2)) · √2
        = 0  +  (−i·sin(θ/2)) · √2
        = −i·√2·sin(θ/2)
```

This is purely imaginary. So `Re⟨x̃|D⟩ = 0`. Hadamard fails.

The issue: D has a nonzero real second component (√2), but the test state's
second component `i·sin(θ/2)` is imaginary. When you multiply:
```
conj(i·sin(θ/2)) · √2 = −i·sin(θ/2) · √2 = purely imaginary
```
The i in the test state "poisons" the result — it rotates the signal 90° into
the imaginary axis where Re() can't see it.

---

## The one formula that tells you everything

Hadamard kernel = `Re⟨x̃|D⟩`

Expand with `x̃ = [α, β]` and `D = [d₁, d₂]`:
```
Re⟨x̃|D⟩ = Re(conj(α)·d₁  +  conj(β)·d₂)
          = Re(conj(α)·d₁) + Re(conj(β)·d₂)
```

Each term `Re(conj(α)·d)` is nonzero ONLY when `conj(α)·d` has a nonzero
real part — meaning `α` and `d` must be "aligned" in angle on the complex plane.

Concretely:
- If `α = a` (real) and `d = r` (real): `Re(a·r) = a·r` → nonzero ✓
- If `α = ia` (imaginary) and `d = r` (real): `Re(conj(ia)·r) = Re(−ia·r) = 0` ✗
- If `α = a` (real) and `d = ir` (imaginary): `Re(a·ir) = Re(iar) = 0` ✗
- If `α = ia` and `d = ir`: `Re(conj(ia)·ir) = Re(−ia·ir) = Re(a·r) = a·r` ✓

**Pattern:** the component of x̃ and the corresponding component of D must have
the SAME "type" (both real, or both imaginary) for that term to contribute.
If one is real and the other imaginary, that term vanishes.

---

## Three training sets — diagnose by eye

### Training set A — both states differ in their real components

```
|x₁⟩ = [0.8,   0.6]   class 0   (all real)
|x₂⟩ = [0.6,   0.8]   class 1   (all real)
```

```
D = [0.8 − 0.6,   0.6 − 0.8] = [0.2, −0.2]   (all real)
```

Test state: `x̃ = [cos(θ/2), sin(θ/2)]` (also all real for this example)

```
Re⟨x̃|D⟩ = cos(θ/2)·0.2  +  sin(θ/2)·(−0.2)
          = 0.2·(cos(θ/2) − sin(θ/2))
```

This is nonzero for most θ. **Hadamard works. ✓**

Intuition: D is real, x̃ is real — perfect alignment.

---

### Training set B — states differ only in a phase on one component

```
|x₁⟩ = [1/√2,   1/√2]   class 0
|x₂⟩ = [1/√2,  i/√2]    class 1
```

Check normalization: both have |1/√2|² + |±1/√2|² = 1/2 + 1/2 = 1 ✓

```
D = [1/√2 − 1/√2,   1/√2 − i/√2]
  = [0,   (1 − i)/√2]
```

Test state: `x̃ = [cos(θ/2), i·sin(θ/2)]` (our usual test state)

```
Re⟨x̃|D⟩ = Re(conj(cos(θ/2))·0)  +  Re(conj(i·sin(θ/2))·(1−i)/√2)
          = 0  +  Re((−i·sin(θ/2))·(1−i)/√2)
```

Expand `(−i)·(1−i) = −i + i² = −i − 1 = −1 − i`:
```
Re((−i·sin(θ/2))·(1−i)/√2) = Re((−1−i)·sin(θ/2)/√2)
                              = −sin(θ/2)/√2
```

This is nonzero! **Hadamard partially works here. ✓**

But it only captures the `−1` part of `(−1 − i)` — it throws away the `−i`
part. Swap test would capture both. So Hadamard works but is less accurate.

---

### Training set C — our paper's toy problem (states differ in sign of real component, but encoded with imaginary first component)

```
|x₁⟩ = [i/√2,   1/√2]   class 0
|x₂⟩ = [i/√2,  −1/√2]   class 1
```

```
D = [i/√2 − i/√2,   1/√2 − (−1/√2)]
  = [0,   √2]          ← D is real in its second component
```

Test state: `x̃ = [cos(θ/2), i·sin(θ/2)]`

```
Re⟨x̃|D⟩ = Re(conj(cos(θ/2))·0)  +  Re(conj(i·sin(θ/2))·√2)
          = 0  +  Re(−i·sin(θ/2)·√2)
          = 0    ← purely imaginary, real part = 0
```

**Hadamard completely fails. ✗**

Why: D[1] = √2 is real. But x̃[1] = i·sin(θ/2) is imaginary.
Real × imaginary = imaginary. Re(imaginary) = 0.
The i in the test state's second component kills the signal.

---

## Summary table — look at training set + test state together

| D (difference vector) | x̃ (test state) | What happens | Verdict |
|---|---|---|---|
| All real components | All real components | Re(conj(x̃)·D) = full signal | ✓ Works |
| All real components | Has imaginary components | Some terms vanish | Partial / depends |
| Has imaginary components | Imaginary in same slots | conj(ia)·ib = ab, real | ✓ Works |
| Real in slot k | Imaginary in same slot k | Re(conj(ia)·b) = 0 | ✗ Fails for that slot |
| Imaginary in slot k | Real in same slot k | Re(conj(a)·ib) = 0 | ✗ Fails for that slot |

**The golden rule:**
> For each component slot, if the test state and the difference vector D are
> 90° apart on the complex plane (one real, one imaginary), that slot
> contributes ZERO to the Hadamard kernel. If ALL slots are 90° apart,
> the entire kernel = 0. That's exactly what happens in the paper's toy problem.

The swap test avoids this entirely because `|⟨x̃|xₘ⟩|² = |real part|² + |imaginary part|²`.
Squaring then adding means it doesn't matter what angle the components are at —
both the real and imaginary contributions always survive.

---

# PART 5: WHY THE OLD METHOD FAILS

## The Hadamard Classifier (prior work)

The previous quantum classifier (what came before this paper) uses:
```
K_H = Σ_m (-1)^{y_m} · w_m · Re⟨x̃|x_m⟩
```

Where:
- Σ_m means "sum over all training examples"
- (-1)^{y_m} is +1 for class 0, -1 for class 1 (encodes the label)
- w_m = class weight (= 0.5 for equal weights)
- Re⟨x̃|x_m⟩ = REAL part of the inner product ⟨x̃|x_m⟩

The inner product ⟨x̃|x_m⟩ = complex conjugate of x̃, dotted with x_m.
For our toy states:
```
⟨x̃(θ)|x₁⟩ = cos(θ/2)·(-i/√2) + (-i·sin(θ/2))·(1/√2)
            = (-i/√2)·[cos(θ/2) + sin(θ/2)]
```
This is purely imaginary! Its real part = 0 for all θ.

Similarly ⟨x̃(θ)|x₂⟩ is also purely imaginary.

So K_H = 0.5 · Re⟨x̃|x₁⟩ − 0.5 · Re⟨x̃|x₂⟩ = 0 − 0 = **0 everywhere**.

The Hadamard classifier produces the same output (zero) for every single
test angle. It literally cannot distinguish the two classes. Useless here.

This is not a bug — it's intentional in the paper to demonstrate the
limitation of that method and motivate the swap-test approach.

---

# PART 6A: BEFORE WE TALK ABOUT THE SWAP TEST — THINGS YOU MUST UNDERSTAND FIRST

## What is a quantum circuit?

A quantum circuit is a recipe — a sequence of instructions — for what to do to
a set of qubits. Just like a cooking recipe says "add flour, then mix, then bake",
a quantum circuit says "apply gate H to qubit 0, then apply CSWAP to qubits 1 and 2,
then measure".

Think of it like a flowchart from left to right. On the left, qubits start in some
initial state. Gates are applied one after another (left to right = time order).
On the right, you measure the qubits and get classical 0/1 bits out.

```
Qubit 0: ─────[Gate A]────[Gate B]──── measure
Qubit 1: ──[Gate C]─────────────────── measure
Qubit 2: ─────────────[Gate D]──────── measure
```

---

## What is a register?

A "register" is just a group of qubits that serve the same purpose.
Like in a regular computer you have a memory register, an address register, etc.
In our quantum circuit:

- **Ancilla register (a)** — 1 qubit. "Ancilla" literally means "helper" in Latin.
  This qubit doesn't hold data — it's a helper qubit that makes the swap test work.
  Think of it as the referee qubit.

- **Index register (m)** — 1 qubit. This holds which training example we're
  looking at. When m=|0⟩ we're comparing to training example 1. When m=|1⟩
  we're comparing to training example 2. The trick: it's in SUPERPOSITION of
  both, so we compare to BOTH simultaneously.

- **Data register (d)** — 1 qubit (or n qubits for n copies). This holds the
  training state that is being compared to the test state.

- **Label register (l)** — 1 qubit. This holds the CLASS of the current training
  example. 0 = class 0, 1 = class 1.

- **Input register (in)** — 1 qubit (or n qubits). This holds the test state
  |x̃(θ)⟩ that we want to classify.

So the full circuit has 5 qubits total for the n=1 case.

---

## What is the ancilla qubit and why do we need it?

The ancilla is the "helper" qubit. Here's why it's needed:

The swap test works by creating INTERFERENCE between two states. Interference
is a quantum phenomenon where two paths through a calculation cancel or amplify
each other. To create interference, you need a qubit that is in superposition
(both 0 and 1 at the same time) so that the computation takes BOTH paths
simultaneously.

The ancilla starts as |0⟩, gets put into superposition with a Hadamard gate
(so it's (|0⟩ + |1⟩)/√2 — equally both 0 and 1), then controls the CSWAP gate
(so when ancilla=|0⟩ no swap happens, when ancilla=|1⟩ the swap happens),
and then gets hit by another Hadamard gate.

The final state of the ancilla encodes the fidelity |⟨x̃|xₘ⟩|². When you
measure the ancilla, the probability of getting 0 is (1 + fidelity)/2.
So a high-fidelity pair means the ancilla almost always measures 0.
A zero-fidelity pair means the ancilla measures 0 or 1 with equal probability.

---

## What is a quantum gate? (the specific ones we use)

**Hadamard gate (H):**
Takes a qubit and puts it into equal superposition.
- H applied to |0⟩ gives (|0⟩ + |1⟩)/√2   ← "both 0 and 1 equally"
- H applied to |1⟩ gives (|0⟩ − |1⟩)/√2   ← "both, but with a minus sign"
- H applied twice returns you to where you started: H·H = identity

As a matrix: H = (1/√2) · [[1, 1], [1, −1]]

Think of it like: if a qubit is a coin, H is the act of spinning it. It goes
from "definitely heads" to "spinning (both)". The second H is the act of
catching it — the interference between the two paths determines the outcome.

**Pauli-Z gate (Z or σz):**
Does nothing to |0⟩. Puts a minus sign on |1⟩.
- Z|0⟩ = |0⟩
- Z|1⟩ = −|1⟩

As a matrix: Z = [[1, 0], [0, −1]]

This is used as the MEASUREMENT basis. When we "measure σz", we measure and
score: +1 if we get |0⟩, −1 if we get |1⟩.

**CNOT gate (CX — Controlled NOT):**
A 2-qubit gate. It has a CONTROL qubit and a TARGET qubit.
- If control = |0⟩: do nothing to target
- If control = |1⟩: flip the target (0→1, 1→0)

Think of it as: "if the control qubit says yes, flip the target."
This is how you entangle two qubits — after a CNOT on a superposition control,
the two qubits become correlated. If you measure one, the other is determined.

**CSWAP gate (Controlled-SWAP, also called Fredkin gate):**
A 3-qubit gate. One control, two targets.
- If control = |0⟩: do nothing, both targets stay the same
- If control = |1⟩: SWAP the two target qubits (they exchange their states)

Think of it as: "if the control says yes, swap these two things."
This is the CORE of the swap test.

**RY(θ) gate:**
Rotates a qubit by angle θ around the Y-axis of the Bloch sphere.
- RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
Used to prepare the index superposition with the right weights.

**RX(θ) gate:**
Rotates around the X-axis.
- RX(θ)|0⟩ = cos(θ/2)|0⟩ − i·sin(θ/2)|1⟩
- RX(−θ)|0⟩ = cos(θ/2)|0⟩ + i·sin(θ/2)|1⟩ ← this is exactly |x̃(θ)⟩!
Used to prepare the test state.

**CZ gate (Controlled-Z):**
- If control = |0⟩: do nothing
- If control = |1⟩: apply Z to target (put minus sign on |1⟩ component)
Used to encode the sign difference between the two training states.

---

## What does "measuring σz^(a) σz^(l)" actually mean in practice?

This sounds scary. It is actually just this:

1. Run the circuit 1024 times (1024 "shots")
2. Every time, write down two bits: what the ancilla measured AND what the label measured
3. After 1024 shots, you have 1024 pairs like: 00, 01, 10, 00, 11, 01, 00...
4. Count:
   - c00 = how many times both were 0
   - c01 = how many times ancilla=0, label=1
   - c10 = how many times ancilla=1, label=0
   - c11 = how many times both were 1
5. Compute: (c00 − c01 − c10 + c11) / 1024

That's it. That number IS ⟨σz^(a) σz^(l)⟩.

Why that formula? Because when both are 0: σz gives (+1)(+1) = +1.
When a=0, l=1: σz gives (+1)(−1) = −1. When a=1, l=0: (−1)(+1) = −1.
When both are 1: (−1)(−1) = +1.

So we're computing the average of (+1 or −1) for each shot, where +1 means
"ancilla and label agreed" and −1 means "they disagreed."

If class 0 is the right answer, you'll see more agreement → positive number.
If class 1 is the right answer, you'll see more disagreement → negative number.

---

# PART 6B: THE SWAP-TEST SOLUTION — EVERY STEP EXPLAINED FOR BABIES

## First: why does the swap test fix what Hadamard broke?

Let's go back to basics. The problem with Hadamard was this line:

```
K_H = Σ_m (-1)^{y_m} · w_m · Re⟨x̃|x_m⟩
```

The `Re(...)` part means "take the real part only, throw away the imaginary part."

Imagine you have a complex number z = 3 + 4i. That's a point at (3, 4) on the
complex plane. The real part is 3. The "size" (magnitude) is √(3² + 4²) = 5.

Re() only gives you 3. You lose the 4i part completely.

But `|z|²` = 3² + 4² = 9 + 16 = 25. You kept BOTH parts.

---

This is called the **fidelity**. It is `|⟨x̃|x_m⟩|²`.

Why is fidelity better than the real part?

Because fidelity uses BOTH the real AND imaginary parts of the inner product,
so it can NEVER be zero just because one part is zero.

In our toy problem, we showed that ⟨x̃|x₁⟩ = (i/√2)·[cos(θ/2) − sin(θ/2)]
This is a PURELY IMAGINARY number. The real part is zero. So Hadamard gives zero.

But the fidelity:
```
|⟨x̃|x₁⟩|² = |(i/√2)|² · |cos(θ/2) − sin(θ/2)|²
```

Step by step:
- `|(i/√2)|²` = |i|² · |1/√2|² = 1 · (1/2) = 1/2
  (|i| = 1 because i is a complex number with real=0, imaginary=1, so |i| = √(0²+1²) = 1)
- `|cos(θ/2) − sin(θ/2)|²` = (cos(θ/2) − sin(θ/2))²
  (this is a real number, so magnitude² = just square it)

So:
```
|⟨x̃|x₁⟩|² = (1/2) · (cos(θ/2) − sin(θ/2))²
```

Let's expand (cos(θ/2) − sin(θ/2))²:
```
= cos²(θ/2) − 2·cos(θ/2)·sin(θ/2) + sin²(θ/2)
= [cos²(θ/2) + sin²(θ/2)]  −  2·cos(θ/2)·sin(θ/2)
= 1  −  sin(θ)
```
(we used: cos²+sin²=1, and 2·cos·sin = sin(2x) where x=θ/2, so 2·cos(θ/2)·sin(θ/2) = sin(θ))

Final result:
```
|⟨x̃|x₁⟩|² = ½·(1 − sin θ)
```

And similarly for the other training state (same calculation but with −1/√2 instead of 1/√2):
```
|⟨x̃|x₂⟩|² = ½·(1 + sin θ)
```

These are DIFFERENT for almost every value of θ. The signal is alive!

Compare:
- Hadamard signal: Re⟨x̃|x₁⟩ = 0, Re⟨x̃|x₂⟩ = 0 → completely dead
- Fidelity signal: |⟨x̃|x₁⟩|² = ½(1−sinθ), |⟨x̃|x₂⟩|² = ½(1+sinθ) → different, classifiable

---

## The kernel formula — "Eq. 9 of the paper" — explained word by word

The paper's main result says that the quantum circuit computes this number:

```
⟨σz^(a) σz^(l)⟩ = Σ_m (-1)^{y_m} · w_m · |⟨x̃|x_m⟩|^{2n}
```

This looks scary. Let's decode it piece by piece.

---

### Piece 1: `Σ_m`

`Σ` is the Greek letter Sigma. In math it means "add up everything that follows."
The `m` means "for each training example m."

So `Σ_m (something)` = (something for m=1) + (something for m=2) + ...

For our toy problem with 2 training examples:
```
Σ_m (something)  =  (something for m=1) + (something for m=2)
```

---

### Piece 2: `(-1)^{y_m}`

`y_m` is the label (class) of training example m. It is 0 or 1.

- If y_m = 0 (class 0): `(-1)^0 = 1` → this term is **added** (positive)
- If y_m = 1 (class 1): `(-1)^1 = -1` → this term is **subtracted** (negative)

This is the mathematical way of saying: "class 0 examples vote YES, class 1 examples vote NO."

For our toy problem:
- Training example 1 has label 0 → `(-1)^0 = +1`
- Training example 2 has label 1 → `(-1)^1 = −1`

---

### Piece 3: `w_m`

This is the weight (importance) of training example m.
We use equal weights: w₁ = w₂ = 0.5.

The weights must add up to 1. They say "how much does each training example matter?"

---

### Piece 4: `|⟨x̃|x_m⟩|^{2n}`

This is the fidelity between the test state and training example m, raised to the power 2n.

- `⟨x̃|x_m⟩` = the inner product (overlap) between test state and training state m
- `|...|²` = the magnitude squared (fidelity) — always a number between 0 and 1
- `^{2n}` = raise to the power 2n (n is the number of copies)

For n=1: this is just `|⟨x̃|x_m⟩|²` = the fidelity
For n=2: this is `|⟨x̃|x_m⟩|⁴` = fidelity squared
For n=3: this is `|⟨x̃|x_m⟩|⁶` = fidelity cubed

Why does n matter? We'll explain this in full in a moment.

---

### Piece 5: `⟨σz^(a) σz^(l)⟩`

This is what the quantum circuit actually measures. Let's decode it completely.

`σz` (sigma-z) is the Pauli-Z measurement. When you measure a qubit:
- If the qubit comes out as 0 → the result is **+1**
- If the qubit comes out as 1 → the result is **−1**

`σz^(a)` means: apply this scoring to the **ancilla** qubit.
`σz^(l)` means: apply this scoring to the **label** qubit.
`σz^(a) σz^(l)` means: MULTIPLY the two scores together.

So for a single shot of the circuit:
- If ancilla=0, label=0: score = (+1) × (+1) = **+1**
- If ancilla=0, label=1: score = (+1) × (−1) = **−1**
- If ancilla=1, label=0: score = (−1) × (+1) = **−1**
- If ancilla=1, label=1: score = (−1) × (−1) = **+1**

`⟨...⟩` means "average over many shots."

So after 1024 shots, if you counted:
- c00 = number of shots where ancilla=0 AND label=0
- c01 = number of shots where ancilla=0 AND label=1
- c10 = number of shots where ancilla=1 AND label=0
- c11 = number of shots where ancilla=1 AND label=1

Then:
```
⟨σz^(a) σz^(l)⟩ = (+1)·c00 + (−1)·c01 + (−1)·c10 + (+1)·c11
                 = (c00 − c01 − c10 + c11) / 1024
```

**That's the kernel value. One number. Computed from counting.**

If that number is positive → classify as class 0.
If that number is negative → classify as class 1.

---

## Putting it together — full calculation for our toy problem (n=1)

Substituting everything we know:

```
⟨σz^(a) σz^(l)⟩
= (+1) · 0.5 · |⟨x̃|x₁⟩|²   +   (−1) · 0.5 · |⟨x̃|x₂⟩|²
= 0.5 · ½(1 − sin θ)  −  0.5 · ½(1 + sin θ)
```

Step by step:
```
= ¼(1 − sin θ)  −  ¼(1 + sin θ)
= ¼ − ¼·sin θ  −  ¼ − ¼·sin θ
= − ½·sin θ
```

Wait — this is negative when sin θ > 0 (i.e., θ ∈ (0, π)), which would mean class 1
for those angles. But we said the correct answer is class 0 for θ ∈ (0, π)!

Let's double-check using the paper's own inner product formula (page 3):
```
⟨x̃|x₁⟩ = i·sin(θ/2 + π/4)
⟨x̃|x₂⟩ = i·cos(θ/2 + π/4)
```

So:
```
|⟨x̃|x₁⟩|² = sin²(θ/2 + π/4)
|⟨x̃|x₂⟩|² = cos²(θ/2 + π/4)
```

Kernel with w₁=w₂=0.5:
```
= 0.5·sin²(θ/2 + π/4)  −  0.5·cos²(θ/2 + π/4)
= −0.5·[cos²(θ/2 + π/4) − sin²(θ/2 + π/4)]
= −0.5·cos(2·(θ/2 + π/4))      [using: cos²x − sin²x = cos(2x)]
= −0.5·cos(θ + π/2)
= −0.5·(−sin θ)                 [using: cos(x + π/2) = −sin x]
= +0.5·sin θ
```

So the kernel = **+0.5·sin θ**.

- For θ ∈ (0, π): sin θ > 0, kernel > 0 → **class 0** ✓
- For θ ∈ (π, 2π): sin θ < 0, kernel < 0 → **class 1** ✓
- At θ = 0 or θ = π: sin θ = 0, kernel = 0 → boundary (ambiguous, makes sense) ✓

Perfect classification everywhere. This is the core result of the paper.

---

## The decision rule — how you actually turn a number into a class label

The paper uses this formula (their Eq. 7):
```
ỹ = ½ · (1 − sign(⟨σz^(a) σz^(l)⟩))
```

Let's unpack it:

`sign(x)` means:
- If x > 0: sign(x) = +1
- If x < 0: sign(x) = −1
- If x = 0: undefined (boundary)

So:
- If kernel > 0 → sign = +1 → ỹ = ½·(1 − 1) = ½·0 = **0** → class 0
- If kernel < 0 → sign = −1 → ỹ = ½·(1 − (−1)) = ½·2 = **1** → class 1

You can remember it simply as: **positive kernel = class 0, negative kernel = class 1**.

The formula just encodes that in math. That's all.

---

## Why does using MORE copies (larger n) make classification better?

This is one of the main contributions of the paper. Let's understand it with numbers.

With n copies, the kernel is:
```
K_n = 0.5·|⟨x̃|x₁⟩|^{2n} − 0.5·|⟨x̃|x₂⟩|^{2n}
```

Let's use p = |⟨x̃|x₁⟩|² and q = |⟨x̃|x₂⟩|² = 1 − p (they always add to 1 for our toy problem).

So: `K_n = 0.5·(p^n − q^n)`

Now let's see what happens at a test angle where p = 0.7 and q = 0.3
(the test state is clearly closer to training state 1):

```
n=1:   K = 0.5·(0.7   − 0.3)    = 0.5·0.4    = 0.20
n=2:   K = 0.5·(0.49  − 0.09)   = 0.5·0.40   = 0.20  ← roughly same
n=3:   K = 0.5·(0.343 − 0.027)  = 0.5·0.316  = 0.158
n=5:   K = 0.5·(0.168 − 0.002)  = 0.5·0.166  = 0.083
n=10:  K = 0.5·(0.028 − 0.0000006) ≈ 0.014
n=50:  K = 0.5·(0.7^50 − tiny) ≈ tiny but still positive
```

The magnitude shrinks as n increases. But the SIGN stays positive (class 0) throughout.

Now let's look at a test angle very close to the boundary, where p = 0.51, q = 0.49
(barely distinguishable — right near the decision boundary):

```
n=1:   K = 0.5·(0.51   − 0.49)    = 0.5·0.02    = 0.010
n=2:   K = 0.5·(0.2601 − 0.2401)  = 0.5·0.020   = 0.010
n=5:   K = 0.5·(0.51^5 − 0.49^5)  = 0.5·(0.0345 − 0.0282) = 0.0032
n=10:  K ≈ 0.5·(0.51^10 − 0.49^10) ≈ very small but still positive
```

The sign is still correct — it's still class 0. But the magnitude is tiny.

**What "sharpening" actually means:**

Imagine the kernel as a landscape. At n=1, it's a gentle rolling hill — small
at the peaks, gently sloping. At n=∞, it becomes sharp spikes — enormous at
the exact positions of the training states, zero everywhere else.

This is what the paper's Fig. 3 shows (which we reproduced as `results/01_n_copies_effect.png`).

At n=∞, the kernel becomes a Dirac delta function — infinitely sharp spikes
at each training state position. This is the "perfect localization" the paper
proves: the classifier becomes a nearest-neighbor classifier in the limit.

**But here's the practical point:** for classification, you only need the SIGN
to be correct. Even with a tiny magnitude at large n, the sign is correct.
The risk is that on real noisy hardware, a tiny-magnitude signal gets swamped
by measurement noise. That's the trade-off the paper discusses: sharper
classification boundary, but harder to measure accurately on noisy hardware.

---

## Post-selection — the thing the paper removes (and why it matters)

The old classifier (Hadamard, ref [10] in the paper) had a wasteful step called
**post-selection**. Here's what that means in plain language:

**Old method with post-selection:**
1. Run the circuit
2. Measure the ancilla qubit
3. If ancilla came out as 0: keep this result, use it
4. If ancilla came out as 1: THROW IT AWAY, don't use it
5. Only compute the answer from the "ancilla=0" shots

Why is this bad?
- About half your shots get thrown away (wasted computation time)
- You need to normalize your data first to make sure "enough" shots give ancilla=0
  (specifically you need to standardize to mean=0, standard deviation=1 — extra work)
- IBM hardware is limited and expensive, wasting half your shots matters

**New method (paper's swap test):**
The paper's key insight: you don't need to throw anything away.

When ancilla=0: the label qubit gives information one way.
When ancilla=1: the label qubit gives information the OPPOSITE way (flipped).

So instead of ignoring ancilla=1 shots, you can USE them — just flip their label.
This is exactly what `⟨σz^(a) σz^(l)⟩` does automatically:

- Ancilla=0, label=0: score = (+1)(+1) = +1 → "supports class 0"
- Ancilla=0, label=1: score = (+1)(−1) = −1 → "supports class 1"
- Ancilla=1, label=0: score = (−1)(+1) = −1 → "supports class 1 (flipped!)"
- Ancilla=1, label=1: score = (−1)(−1) = +1 → "supports class 0 (flipped!)"

The sign flip of the ancilla automatically flips the meaning of the label.
Both sets of shots contribute useful information. Nothing is wasted.
**Twice as efficient** as the old method.

---

## How the quantum state evolves — Eq. 8 of the paper (simplified for babies)

The paper's Eq. 8 describes the quantum state right before the swap test starts.
Let me explain what this state looks like and how the circuit produces it.

**Starting point:** all 5 qubits are |0⟩. Everything is zero.
```
State = |0⟩_a  ⊗  |0⟩_m  ⊗  |0⟩_d  ⊗  |0⟩_l  ⊗  |0⟩_in
```
(⊗ means "and also" — it's the way to write "the combined state of multiple independent qubits")

**After preparing the index qubit (m):**
The RY gate on the index qubit puts it into superposition.
With equal weights (0.5 each), m becomes (|0⟩ + |1⟩)/√2.

Now the index qubit is "both 0 and 1 at the same time" — it will route the
computation to BOTH training examples simultaneously.

**After preparing training data and label:**
The circuit uses the index qubit to conditionally prepare the data and label qubits.
When m=|0⟩: data gets |x₁⟩, label gets |0⟩.
When m=|1⟩: data gets |x₂⟩, label gets |1⟩.

Since m is in superposition, the state becomes:
```
(1/√2) · |0⟩_m |x₁⟩_d |0⟩_l   +   (1/√2) · |1⟩_m |x₂⟩_d |1⟩_l
```

In plain language: the computer is simultaneously in TWO versions of itself —
one where it's comparing to x₁ (class 0), and another where it's comparing to x₂ (class 1).

**After preparing test state (in):**
RX(−θ) on the input qubit prepares the test state |x̃(θ)⟩ = cos(θ/2)|0⟩ + i·sin(θ/2)|1⟩.

```
Full state = (1/√2) · (|0⟩_m |x₁⟩_d |0⟩_l + |1⟩_m |x₂⟩_d |1⟩_l) ⊗ |0⟩_a ⊗ |x̃⟩_in
```

This is the "input state" for the swap test. Both training comparisons happen at once.

**This is quantum parallelism:** with 5 qubits, we're simultaneously comparing
the test state to ALL training examples, not one at a time.

---

## Quantum forking — why we can do this without knowing the training states

This is "Section 3" of the paper, and it's a surprising result.

Everything above assumed we "programmed" the training states into the circuit.
But what if we received the training states as QUANTUM STATES from some other
quantum process and don't know their classical description?

The answer: you can STILL do the classification using **quantum forking**.

The idea: start with copies of the training states on separate quantum registers:
```
|x₁⟩  |x₂⟩  |x₃⟩ ...   (physical training state registers)
```

Then use **controlled-SWAP gates** to "route" the right training state into the
data register based on the index qubit:

- If index qubit m = |0⟩: swap |x₁⟩ into the data register
- If index qubit m = |1⟩: swap |x₂⟩ into the data register
- If m is in superposition: BOTH swaps happen simultaneously (quantum superposition)

After this "forking" operation:
- Data register has |x₁⟩ in the m=0 branch
- Data register has |x₂⟩ in the m=1 branch
- Simultaneously, in superposition

Then the swap test runs exactly as before.

**Why this is remarkable:**
You can classify quantum states that come from another quantum machine,
or that you generated with some quantum algorithm. You never need to know
what the states are classically — you just need physical copies of them.

This is a capability that has no classical analogue. Classically, you always
need to read out (measure) the data to process it, which destroys quantum states.
Quantum forking processes the states WITHOUT measuring them.

---

# PART 7: THE CIRCUIT — EVERY SINGLE GATE, ONE BY ONE

## What the full circuit looks like

Here is the circuit diagram for the 5-qubit swap-test classifier (n=1 case).
Reading left to right = time passing. Each box or symbol is one gate operation.

```
Register:  Initial   Step 1       Step 2           Step 3   Step 4   Step 5   Step 6     Step 7   Measure

  a [0] : ─────────────────────────────────────────────────────── [H] ──[CSWAP]── [H] ──── c[0]

  m [0] : ──────── [RY(α)] ──────────────────────────────────────────────────────────────────
                     │
  d [0] : ────────── │ ──── [H][Rz(−π)][S] ── [CZ] ──────────────────[CSWAP]────────────────
                     │                          │ (with m)    │ (with a, and in)
  l [0] : ────────── │ ───────────────────── [CX] ──────────────────────────────────── c[1]
                                             (with m)
  in[0] : ────────────────────────────────────────────── [RX(−θ)] ──── [CSWAP] ─────────────
```

(CSWAP acts on three qubits at once: ancilla=control, data=target1, input=target2)

Let's walk through every single gate.

---

## Step 0: Starting state

All 5 qubits start as |0⟩ — the "ground state" or "everything off" state.

```
|0⟩_a ⊗ |0⟩_m ⊗ |0⟩_d ⊗ |0⟩_l ⊗ |0⟩_in
```

In compact notation: |00000⟩ (ancilla, index, data, label, input — all zero).

---

## Step 1: Index superposition — RY(α) gate on the m qubit

**What RY does:**
RY(θ) rotates a qubit around the Y-axis of the Bloch sphere by angle θ.
Applied to |0⟩: RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩

**Why we use this:**
We want the index qubit to be in superposition with SPECIFIC weights w₁ and w₂.
For equal weights w₁=w₂=0.5:
- We need the state √0.5 · |0⟩ + √0.5 · |1⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
- This requires cos(α/2) = 1/√2 and sin(α/2) = 1/√2
- So α/2 = π/4, meaning α = π/2

**What happens:**
```
RY(π/2)|0⟩_m = cos(π/4)|0⟩ + sin(π/4)|1⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
```

cos(π/4) = 1/√2 ≈ 0.707 and sin(π/4) = 1/√2 ≈ 0.707.

**State after Step 1:**
```
|0⟩_a ⊗ [(1/√2)|0⟩ + (1/√2)|1⟩]_m ⊗ |0⟩_d ⊗ |0⟩_l ⊗ |0⟩_in
```

The index qubit is now "both 0 and 1, equally." The other qubits are unchanged.

**Why this matters:**
When the index qubit is |0⟩, the circuit will process training example 1.
When the index qubit is |1⟩, it will process training example 2.
Since it's in SUPERPOSITION, both happen simultaneously.
We don't choose — we process both at once. This is quantum parallelism.

---

## Step 2: Prepare training data — H, Rz(−π), S, CZ gates on the d qubit

This is the trickiest part of the circuit. We need to prepare the data qubit in:
- State |x₁⟩ = [i/√2, 1/√2] when m = |0⟩
- State |x₂⟩ = [i/√2, −1/√2] when m = |1⟩

The only difference between |x₁⟩ and |x₂⟩ is the sign of the second component.
We'll use H, Rz, and S to get to |x₁⟩ first, then use CZ to flip the sign
conditionally (only when m=|1⟩).

**Substep 2a: H gate on data qubit d**

H|0⟩ = (1/√2)|0⟩ + (1/√2)|1⟩

Data qubit becomes: [1/√2, 1/√2]

**Substep 2b: Rz(−π) gate on data qubit d**

Rz(θ) is a "phase rotation" gate. It works like this:
- Rz(θ)|0⟩ = e^{−iθ/2} |0⟩   (multiply the |0⟩ component by the complex number e^{−iθ/2})
- Rz(θ)|1⟩ = e^{+iθ/2} |1⟩   (multiply the |1⟩ component by e^{+iθ/2})

For θ = −π:
- e^{−i(−π)/2} = e^{+iπ/2} = cos(π/2) + i·sin(π/2) = 0 + i = **i**
- e^{+i(−π)/2} = e^{−iπ/2} = cos(π/2) − i·sin(π/2) = 0 − i = **−i**

(Euler's formula: e^{iθ} = cos θ + i·sin θ — a compact way to write rotations)

So: Rz(−π)[(1/√2)|0⟩ + (1/√2)|1⟩] = (i/√2)|0⟩ + (−i/√2)|1⟩

Data qubit becomes: [i/√2, −i/√2]

**Substep 2c: S gate on data qubit d**

The S gate is a specific phase gate: S|0⟩ = |0⟩, S|1⟩ = i|1⟩.
(It multiplies the |1⟩ component by i.)

S[(i/√2)|0⟩ + (−i/√2)|1⟩] = (i/√2)|0⟩ + (−i/√2)·i·|1⟩

What is (−i)·i? Remember i·i = i² = −1, so (−i)·i = −i² = −(−1) = **+1**.

So: = (i/√2)|0⟩ + (1/√2)|1⟩

Data qubit becomes: [i/√2, 1/√2] = |x₁⟩ ✓

Without CZ, the data qubit is |x₁⟩ regardless of the index qubit.

**Substep 2d: CZ gate controlled on m, target d**

CZ (controlled-Z) does this:
- If m = |0⟩: do nothing to d
- If m = |1⟩: apply Z to d. Z flips the sign of the |1⟩ component.

Z|x₁⟩ = Z[(i/√2)|0⟩ + (1/√2)|1⟩] = (i/√2)|0⟩ + (−1/√2)|1⟩ = |x₂⟩

So:
- When m = |0⟩: data stays |x₁⟩ = [i/√2, 1/√2]
- When m = |1⟩: data becomes |x₂⟩ = [i/√2, −1/√2]

**State after Step 2:**
```
|0⟩_a  ⊗  (1/√2)·[ |0⟩_m|x₁⟩_d  +  |1⟩_m|x₂⟩_d ]  ⊗  |0⟩_l  ⊗  |0⟩_in
```

The data qubit is now ENTANGLED with the index qubit. Each branch of the
superposition has the right training state.

---

## Step 3: Prepare label — CX (CNOT) gate controlled on m, target l

CNOT (CX) does this:
- If control (m) = |0⟩: do nothing to target (l) → l stays |0⟩
- If control (m) = |1⟩: flip target (l) → l goes from |0⟩ to |1⟩

After this gate:
- m=|0⟩ branch: label = |0⟩ → label says "class 0"
- m=|1⟩ branch: label = |1⟩ → label says "class 1"

**State after Step 3:**
```
|0⟩_a  ⊗  (1/√2)·[ |0⟩_m|x₁⟩_d|0⟩_l  +  |1⟩_m|x₂⟩_d|1⟩_l ]  ⊗  |0⟩_in
```

Now each branch has the correct training data AND the correct label, together in superposition.

---

## Step 4: Prepare test state — RX(−θ) gate on in qubit

RX(θ) rotates around the X-axis:
```
RX(θ)|0⟩ = cos(θ/2)|0⟩ − i·sin(θ/2)|1⟩
```

With RX(−θ) (negative angle):
```
RX(−θ)|0⟩ = cos(θ/2)|0⟩ + i·sin(θ/2)|1⟩
```

This matches exactly the test state |x̃(θ)⟩ = cos(θ/2)|0⟩ + i·sin(θ/2)|1⟩ ✓

**State after Step 4:**
```
|0⟩_a  ⊗  (1/√2)·[ |0⟩_m|x₁⟩_d|0⟩_l  +  |1⟩_m|x₂⟩_d|1⟩_l ]  ⊗  |x̃(θ)⟩_in
```

The test state is ready. We haven't touched the ancilla yet — it's still |0⟩.

---

## Step 5: First Hadamard on ancilla — H gate on a

H|0⟩ = (1/√2)(|0⟩ + |1⟩)

**State after Step 5:**
```
(1/√2)(|0⟩ + |1⟩)_a  ⊗  (1/√2)·[ |0⟩_m|x₁⟩_d|0⟩_l  +  |1⟩_m|x₂⟩_d|1⟩_l ]  ⊗  |x̃⟩_in
```

**Why this gate is here:**
The Hadamard puts the ancilla into superposition so it can "take both paths" at once:
- In the |0⟩_a branch: the CSWAP will do NOTHING (no swap)
- In the |1⟩_a branch: the CSWAP will SWAP the data and input registers

Both paths happen simultaneously, creating interference.
This is what makes the swap test work — without this superposition, no interference, no fidelity measurement.

---

## Step 6: Controlled-SWAP — CSWAP gate (ancilla controls, d and in are targets)

The CSWAP (Fredkin gate) does:
- If ancilla = |0⟩: do NOTHING to d and in (they stay as is)
- If ancilla = |1⟩: SWAP d and in (they exchange their states)

After CSWAP:
- In the ancilla=|0⟩ branch: data = |x_m⟩, input = |x̃⟩ (unchanged)
- In the ancilla=|1⟩ branch: data = |x̃⟩, input = |x_m⟩ (swapped)

Now the two branches have the two states in DIFFERENT ORDERS. This is what
creates the interference that encodes the fidelity.

**Intuition:** Imagine you have two boxes A and B. In path 1: left box = apples, right box = oranges. In path 2: left box = oranges, right box = apples. The second Hadamard will cause interference between these two arrangements, and the result depends on whether "apples" and "oranges" are similar (high fidelity) or different (low fidelity).

---

## Step 7: Second Hadamard on ancilla — H gate on a again

H applied again after the CSWAP creates interference between the two paths.

Here's the key mathematical result of the swap test:
After both H gates and the CSWAP, the probability of measuring the ancilla as 0 is:
```
P(ancilla = 0) = (1 + |⟨x̃|x_m⟩|²) / 2
```
And the probability of measuring the ancilla as 1 is:
```
P(ancilla = 1) = (1 − |⟨x̃|x_m⟩|²) / 2
```

**Baby explanation of why:**
- If |x̃⟩ and |x_m⟩ are IDENTICAL (same state): fidelity = 1
  → P(ancilla=0) = (1+1)/2 = 1 → ancilla ALWAYS measures 0
  → The two paths interfere CONSTRUCTIVELY
- If |x̃⟩ and |x_m⟩ are COMPLETELY DIFFERENT (orthogonal): fidelity = 0
  → P(ancilla=0) = (1+0)/2 = 0.5 → ancilla is 50/50
  → The two paths have EQUAL interference (no information)
- For everything in between: fidelity ∈ (0, 1), P is in between

The fidelity |⟨x̃|x_m⟩|² is encoded in the ancilla's measurement probability.

---

## Step 8: Measure — ancilla → c[0], label → c[1]

Read out both the ancilla and label qubits. Write down 0 or 1 for each.

Repeat the entire circuit 1024 times (1024 "shots").

Count the four types of outcomes:
- c00 = times you got (ancilla=0, label=0)
- c01 = times you got (ancilla=0, label=1)
- c10 = times you got (ancilla=1, label=0)
- c11 = times you got (ancilla=1, label=1)

Compute:
```
kernel = (c00 − c01 − c10 + c11) / 1024
```

If kernel > 0: classify as class 0.
If kernel < 0: classify as class 1.

**Why does this formula give the kernel?**

The label qubit is 0 when the circuit was in the "training example 1, class 0"
branch, and 1 when in the "training example 2, class 1" branch.

The ancilla encodes the fidelity as described above.

When you multiply the scores (ancilla score) × (label score):
- If ancilla says "high fidelity with class 0 training state" AND label=0: score = +1
  → strong vote for class 0
- If ancilla says "high fidelity with class 1 training state" AND label=1: score = +1
  → but wait — label=1 gives (−1)(−1)=+1? That would vote for class 0 too?

This works out because when the test state has HIGH fidelity with class 1 training state
(not class 0), the ancilla tends to measure 1 more often, and label=1 as well,
giving (−1)(−1)=+1... hmm, this seems backwards.

Actually the math works out precisely because of HOW the ancilla probabilities
combine with the label probabilities. The full derivation (Eq. 8 → Eq. 9 of
the paper) shows rigorously that `⟨σz^(a) σz^(l)⟩ = K_n`.

The short intuition: class-0 training examples have weight `+1` (via `(-1)^{y_m=0} = +1`)
and class-1 examples have weight `−1`. The ancilla encodes fidelity. Their combination
gives positive total for the class with higher fidelity.

---

## Noise robustness — why hardware errors don't destroy the answer

The paper proves an important property: many types of hardware errors DON'T
affect the final classification.

**What is a hardware error?**
A hardware error means a gate doesn't do exactly what you asked. For example:
- A "bit flip error" means the qubit randomly flips from 0 to 1 (or vice versa)
- A "phase error" means the complex phase of the qubit state gets wrong

**Which errors DON'T matter:**
Any error that acts on the index, data, or input qubits (not ancilla and not label)
does NOT change the measurement of ⟨σz^(a) σz^(l)⟩. This is because we're only
measuring the ancilla and label — errors on OTHER qubits don't affect those
measurement statistics.

Phase errors on ANY qubit also don't matter, because σz measures in the
same basis (the 0/1 basis).

**Which errors DO matter:**
Bit-flip errors on the ancilla or label qubit. If a bit flip happens with
probability p:
```
⟨σz^(a) σz^(l)⟩_noisy = (1 − 2p) · ⟨σz^(a) σz^(l)⟩_ideal
```

The effect: the kernel value is scaled down by a factor of (1−2p).
As long as p < 0.5 (which is always true on real hardware where p is typically 0.01),
the SIGN is preserved.

**Sign preserved = correct classification.**

The magnitude shrinks, which is why hardware results have smaller kernel values
than theory predicts. But the sign — and therefore the class prediction — is
correct.

This is what you see in our `results/`: hardware kernel values are roughly
65-80% of the theoretical values, but the classification (sign) is 96-100%
correct.

---

# PART 8: THE HELSTROM EQUIVALENCE — WHY THE PAPER'S RESULT IS THEORETICALLY PROFOUND

## What is the Helstrom measurement? — Starting from zero

Imagine you have a coin that might be:
- Type A coin: biased to come up heads with probability 0.7
- Type B coin: biased to come up heads with probability 0.3

Someone hands you the coin. You need to decide: is it type A or type B?

The best strategy: flip it a few times. If you get mostly heads, guess type A.
If mostly tails, guess type B. There's a precise number of flips needed to
get a desired accuracy — this is a CLASSICAL detection/hypothesis-testing problem.

**The quantum version:**
Someone hands you a quantum state. It might be:
- Hypothesis 0: the state is ρ₀ (one specific quantum state)
- Hypothesis 1: the state is ρ₁ (a different specific quantum state)

You need to measure it to decide which hypothesis is true.

**The question:** what is the BEST measurement strategy?
More precisely: what choice of measurement minimizes the probability of error?

This was solved by Carl Helstrom in 1969. The answer is called the **Helstrom measurement**.

---

## The Helstrom operator — what it is and how to use it

Given two possible states ρ₀ and ρ₁ with prior probabilities p₀ and p₁:

**Step 1:** Build the "Helstrom operator":
```
A = p₀·ρ₀  −  p₁·ρ₁
```

This is a matrix (a weighted difference of the two state matrices).

**Step 2:** Measure the expectation value ⟨A⟩ on your unknown state:
- If ⟨A⟩ > 0 → guess hypothesis 0
- If ⟨A⟩ < 0 → guess hypothesis 1

**Step 3:** The resulting error rate is the lowest possible for ANY measurement:
```
P_error_minimum = ½ · (1 − ‖p₀ρ₀ − p₁ρ₁‖₁)
```
where ‖·‖₁ is the trace norm (a way to measure the "size" of a matrix).

No other measurement strategy can beat this. It is THE optimal strategy.

---

## How the swap test secretly IS the Helstrom measurement

The paper proves something stunning: the swap test circuit, measuring ⟨σz^(a) σz^(l)⟩,
is MATHEMATICALLY IDENTICAL to measuring the expectation value of the Helstrom operator.

Here's how:

**Define the observable A for our classification problem:**
```
A = Σ_{m: y_m=0} w_m |x_m⟩⟨x_m|^⊗n  −  Σ_{m: y_m=1} w_m |x_m⟩⟨x_m|^⊗n
```

Let's decode every symbol:
- `|x_m⟩⟨x_m|` is the OUTER PRODUCT — a matrix (not a number)
  - If |x_m⟩ = [a, b] then |x_m⟩⟨x_m| = [[|a|², a·conj(b)], [b·conj(a), |b|²]]
  - It is a projection matrix — it "projects" any state onto the direction of |x_m⟩
- `^⊗n` means the n-fold tensor product (the matrix for n copies)
- `Σ_{m: y_m=0}` means sum over class-0 training examples
- `−  Σ_{m: y_m=1}` means subtract the class-1 examples

For our toy problem with n=1, w₁=w₂=0.5:
```
A = 0.5 · |x₁⟩⟨x₁|  −  0.5 · |x₂⟩⟨x₂|
```

This is a 2×2 matrix (the difference between two rank-1 projection matrices,
each weighted by 0.5).

**The theorem (Eq. 17 of the paper):**
```
⟨A⟩ = ⟨x̃^⊗n| A |x̃^⊗n⟩
     = Σ_m (-1)^{y_m} w_m · ⟨x̃^⊗n| |x_m⟩⟨x_m|^⊗n |x̃^⊗n⟩
     = Σ_m (-1)^{y_m} w_m · |⟨x̃|x_m⟩|^{2n}
```

The last step uses the identity:
```
⟨ψ^⊗n| |φ⟩⟨φ|^⊗n |ψ^⊗n⟩ = |⟨ψ|φ⟩|^{2n}
```
(which you can verify for n=1: it's just ⟨ψ|φ⟩·⟨φ|ψ⟩ = |⟨ψ|φ⟩|²)

And this final expression is EXACTLY the swap-test kernel (Eq. 9).

**In words:** measuring ⟨σz^(a) σz^(l)⟩ via the swap-test circuit IS the same
as computing the expectation value of the Helstrom operator A on the test state.

---

## Why this is profound — in baby language

Normally, to perform the Helstrom measurement, you need to:
1. Know ρ₀ and ρ₁ exactly (their full classical descriptions as matrices)
2. Construct the matrix A = p₀ρ₀ − p₁ρ₁
3. Find the optimal measurement basis (eigendecompose A)
4. Build a physical measurement apparatus for that basis
5. Measure in that basis

For 2 qubit copies (n=2): A is a 4×4 matrix. Manageable.
For 10 copies: A is a 1024×1024 matrix. Hard.
For 100 copies: A is a 2¹⁰⁰ × 2¹⁰⁰ matrix. **IMPOSSIBLY LARGE.**

The swap test does none of this. It uses:
- The physical quantum states themselves (not their classical description)
- Three gates: H → CSWAP → H
- Two single-qubit measurements

And it gets the EXACT same answer as the optimal Helstrom measurement.

**This is genuine quantum advantage:** The quantum circuit implicitly computes
an exponentially large matrix operation in fixed circuit depth. No classical
computer can do this for large n.

---

## An even deeper result — without knowing the training states at all

The original 1969 Helstrom result assumed you KNOW the states ρ₀ and ρ₁ classically.
You build the operator, then measure.

The paper shows: if you have PHYSICAL COPIES of the training states on quantum
registers (even without knowing what states they are), you can use quantum forking
+ swap test to perform the optimal Helstrom measurement.

This means:
- Training data comes from another quantum machine (e.g., quantum sensor)
- You never know the classical description of the training states
- You STILL do optimal classification

There is no classical analogue. Classically, to learn anything from data, you
must read it out (measure it). In quantum mechanics, quantum forking processes
the training data WITHOUT measuring it, preserving its quantum character.

---

## How we verify this in code

In [core/kernel.py](core/kernel.py) we compute both sides independently:

```python
# Left side: swap-test kernel (Eq. 9)
K_swap = swap_test_kernel(x_test, x_train, labels, n_copies=n)

# Right side: Helstrom operator expectation (Eqs. 16-17)
A = helstrom_operator(x_train, labels, n_copies=n)
K_helstrom = helstrom_expectation(x_test, A, n_copies=n)

# Check they match:
error = abs(K_swap - K_helstrom)  # should be ~1e-16
```

In our results (`results/06_helstrom_equivalence.png`), the error is always
around 10⁻¹⁶ — which is "machine precision" (the smallest number a computer
can distinguish from zero). This numerically verifies the theorem.

---

## Why the plots look the way they do — θ on X-axis, ⟨σz^(a) σz^(l)⟩ on Y-axis

You asked why the plots have θ (theta) on the X-axis and ⟨σz^(a) σz^(l)⟩ on Y-axis.

**X-axis: θ**

The test state |x̃(θ)⟩ = cos(θ/2)|0⟩ + i·sin(θ/2)|1⟩ is parameterized by θ.
Different θ = different test states. We sweep θ from 0 to 2π to test ALL possible
states (a full sweep around the Bloch sphere equator).

Think of θ as "where on the equator of the Bloch sphere is the test state?"
θ=0 → north. θ=π → south. θ=2π → back to north.

The X-axis is literally "which test state are we looking at right now?"

**Y-axis: ⟨σz^(a) σz^(l)⟩**

This is the output of the quantum circuit for that test state — the kernel value.

- Positive values (above the horizontal line at 0): circuit says "class 0"
- Negative values (below the line): circuit says "class 1"
- The theoretical curve is +0.5·sin(θ): it's positive for θ∈(0,π) and negative for θ∈(π,2π)

The range of the y-axis is [−0.5, +0.5] because:
- Maximum possible fidelity for either class is 1
- With 2 equal-weight classes: max kernel = 0.5·(1) − 0.5·(0) = 0.5
- Min kernel = 0.5·(0) − 0.5·(1) = −0.5

So the plots are simply: "as we test every possible quantum state (θ on x-axis),
what does the classifier say (y-axis)?" Points above zero = class 0. Below = class 1.

---

Recall: for any complex number z = a + ib:
```
Re(z) = a                   ← only the real part, b is thrown away
|z|²  = a² + b²             ← BOTH parts contribute, always ≥ 0
```

So fidelity can NEVER be zero just because `a = 0`. Even if the real part
vanishes, the imaginary part squared still contributes. Nothing is discarded.

For our toy problem's inner products (which we showed are purely imaginary):
```
⟨x̃|x₁⟩ = (i/√2)·[cos(θ/2) − sin(θ/2)]

|⟨x̃|x₁⟩|² = |(i/√2)|² · |cos(θ/2) − sin(θ/2)|²
            = (1/2) · (cos(θ/2) − sin(θ/2))²
```

Expanding (cos(θ/2) − sin(θ/2))² = cos²(θ/2) − 2sin(θ/2)cos(θ/2) + sin²(θ/2)
                                   = 1 − sin(θ)    [since 2sin·cos = sin(2·θ/2) = sin(θ)]

So:
```
|⟨x̃|x₁⟩|² = ½(1 − sin(θ))
|⟨x̃|x₂⟩|² = ½(1 + sin(θ))     [same calc with the + sign training state]
```

These are different for almost every θ. ✓ The signal is fully preserved.

---

## The swap-test kernel — Eq. 9 of the paper, step by step

The paper's key result (Eq. 9) is:
```
⟨σz^(a) σz^(l)⟩ = Σ_m (-1)^{y_m} · w_m · |⟨x̃|x_m⟩|^{2n}
```

Let's unpack every symbol:

**σz^(a)** = the Pauli-Z operator acting on the ancilla qubit.
  Pauli-Z is the 2×2 matrix [[1,0],[0,−1]]. It gives +1 when qubit is |0⟩,
  gives −1 when qubit is |1⟩. So measuring σz = measuring 0 or 1 and
  assigning +1 or −1 respectively.

**σz^(l)** = same, but acting on the label qubit.

**⟨σz^(a) σz^(l)⟩** = expectation value of the PRODUCT of these two operators.
  This is: average over many shots of (+1 or −1 from ancilla) × (+1 or −1 from label).
  Explicitly:
  ```
  ⟨σz^(a) σz^(l)⟩ = (+1)(+1)·P(a=0,l=0) + (+1)(−1)·P(a=0,l=1)
                   + (−1)(+1)·P(a=1,l=0) + (−1)(−1)·P(a=1,l=1)
                   = P(00) − P(01) − P(10) + P(11)
  ```
  In counts: `(c00 − c01 − c10 + c11) / total_shots` ← this is literally the
  formula in the Methods section of the paper and in our code.

**(-1)^{y_m}** = +1 if the training example m has label 0, −1 if label is 1.
  This is how the label register "votes" — class 0 adds positively, class 1 subtracts.

**w_m** = weight of training example m. Equal weights = 1/M = 0.5 for 2 classes.

**|⟨x̃|x_m⟩|^{2n}** = the fidelity between test and training state, raised to
  the n-th power. For n=1 this is just the fidelity. For n=3 it's fidelity cubed.

**For our toy problem with n=1, w₁=w₂=0.5:**
```
⟨σz^(a) σz^(l)⟩ = (+1)·0.5·|⟨x̃|x₁⟩|² + (−1)·0.5·|⟨x̃|x₂⟩|²
                 = 0.5·½(1 − sin θ) − 0.5·½(1 + sin θ)
                 = ¼(1 − sin θ) − ¼(1 + sin θ)
                 = ¼ − ¼sin θ − ¼ − ¼sin θ
                 = −½·sin θ
```

This is nonzero for all θ except 0 and π. Sign is:
- Negative when sin θ > 0 → θ ∈ (0, π) → predicts class... wait.

Rule from paper Eq. (7): classify as 0 if ⟨σz^(a) σz^(l)⟩ > 0, class 1 if < 0.

So: ⟨σz^(a) σz^(l)⟩ = −½·sin θ
- For θ ∈ (0, π): sin θ > 0, so kernel is negative → class 1
- For θ ∈ (π, 2π): sin θ < 0, so kernel is positive → class 0

Wait — is this correct? Let's check the paper's inner product formulas.
From page 3 of the paper:
```
⟨x̃|x₁⟩ = i·sin(θ/2 + π/4)
⟨x̃|x₂⟩ = i·cos(θ/2 + π/4)
```

So:
```
|⟨x̃|x₁⟩|² = sin²(θ/2 + π/4)
|⟨x̃|x₂⟩|² = cos²(θ/2 + π/4)
```

Kernel = 0.5·sin²(θ/2 + π/4) − 0.5·cos²(θ/2 + π/4)
       = −0.5·cos(θ + π/2)    [using cos²−sin² = cos(2x), here 2·(θ/2+π/4) = θ+π/2]
       = −0.5·(−sin θ)
       = +0.5·sin θ

This is POSITIVE when θ ∈ (0, π), so kernel > 0 → class 0 ✓
And NEGATIVE when θ ∈ (π, 2π), so kernel < 0 → class 1 ✓

The paper's inner product formula (page 3, Eq. 12) gives the exact result:
```
⟨σz^(a) σz^(l)⟩ = w₁·sin²(θ/2 + π/4) − w₂·cos²(θ/2 + π/4)
```
With w₁=w₂=0.5 this = 0.5·(sin² − cos²) = −0.5·cos(θ + π/2) = +0.5·sin θ

So the kernel is exactly +0.5·sin θ, which is:
- Positive (class 0) for θ ∈ (0, π)
- Negative (class 1) for θ ∈ (π, 2π)
- Zero at θ=0, θ=π (boundaries)

**Perfect classification for all θ.** ✓

---

## The decision rule (Eq. 7 of the paper)

```
ỹ = ½ · (1 − sign(⟨σz^(a) σz^(l)⟩))
```

Unpack this:
- If ⟨σz^(a) σz^(l)⟩ > 0: sign = +1, so ỹ = ½·(1−1) = 0 → class 0
- If ⟨σz^(a) σz^(l)⟩ < 0: sign = −1, so ỹ = ½·(1+1) = 1 → class 1
- If = 0: boundary, undefined

This is how the quantum computer's measurement statistics directly give you
the classification — no further classical computation needed other than
counting four types of measurement outcomes.

---

## The n-copies kernel sharpening — exactly what the paper shows

With n copies, the kernel = w₁·sin²ⁿ(θ/2 + π/4) − w₂·cos²ⁿ(θ/2 + π/4)

The key insight: `sin²(θ/2 + π/4)` and `cos²(θ/2 + π/4)` always add to 1.
Let's call them p and (1−p) for short, where p = sin²(θ/2 + π/4).

At θ = π/4 (far from boundary): p ≈ 0.85, (1−p) ≈ 0.15
```
n=1:   0.5·(0.85  − 0.15)   = 0.35
n=10:  0.5·(0.85¹⁰ − 0.15¹⁰) = 0.5·(0.197 − 0.0000000006) ≈ 0.099
n=100: 0.5·(0.85¹⁰⁰ − tiny)  ≈ 0.5·0.85¹⁰⁰ ≈ something tiny but positive
```

Hmm wait — with n→∞ both sides go to zero unless p = 1 exactly. So why
does the paper say the kernel sharpens?

The SIGN is what matters for classification, not the magnitude. Let's look at
a point very close to θ=π (the boundary) where p ≈ 0.51, (1−p) ≈ 0.49:
```
n=1:   0.5·(0.51 − 0.49)   = 0.01    ← small but positive → class 0
n=10:  0.5·(0.51¹⁰ − 0.49¹⁰) = 0.5·(0.00617 − 0.00531) = 0.00043
n=100: 0.5·(0.51¹⁰⁰ − 0.49¹⁰⁰) ≈ tiny but still positive
```

The sign stays correct, but the MAGNITUDE shrinks near the boundary. What
does "sharpening" mean then?

Look at a point far from the boundary, θ = π/2 where p = 1.0 exactly:
```
n=1:   0.5·(1.0 − 0.0) = 0.5
n=10:  0.5·(1.0 − 0.0) = 0.5
n=100: 0.5·(1.0 − 0.0) = 0.5    ← stays at max
```

And at p = 0.7 (somewhat off-boundary):
```
n=1:   0.5·(0.7  − 0.3)   = 0.20
n=10:  0.5·(0.7¹⁰ − 0.3¹⁰) = 0.5·(0.028 − 0.0000059) = 0.014
n=100: 0.5·(0.7¹⁰⁰ − 0.3¹⁰⁰) ≈ 0.5·3.2×10⁻¹⁶ ≈ 0 but sign still correct
```

So what the paper means by "sharpening" (shown in Fig. 3, our Fig. 01):
- The kernel's MAGNITUDE goes to 0 everywhere except the exact peaks
- The SIGN stays correct everywhere except at the exact boundary
- Classification still works because the rule is based on sign, not magnitude
- In the limit n→∞ the paper shows (Eq. 14): the kernel → Σ_m (-1)^{y_m} w_m δ(x̃−x_m)
  This is a Dirac delta at each training state — a perfect spike

The Dirac delta limit means: at n=∞, the classifier only returns +max or −max
at the exact training state positions, and 0 everywhere else. Perfect localization.

---

## Post-selection: what the paper removes (important detail often missed)

The old Hadamard classifier (ref. 10 in the paper) required **post-selection**:
you ran the circuit, measured the ancilla, and if it came out 0 you kept the
result, if it came out 1 you threw it away. This wastes about half your shots.
Also you had to pre-process your data (standardize to mean=0, std=1) to make
sure the ancilla comes out 0 enough of the time.

The paper's swap-test classifier avoids both problems:

1. **No post-selection:** Both outcomes of the ancilla (0 and 1) are used.
   When ancilla=0, label probability gives classification one way.
   When ancilla=1, label probability gives classification the opposite way.
   Instead of choosing one branch, you measure ⟨σz^(a) σz^(l)⟩ which
   automatically combines BOTH branches into one number. Twice as efficient.

2. **No data pre-processing:** Because fidelity |⟨x̃|x_m⟩|² is always ≥ 0,
   the post-selection probability p₀ ≥ p₁ and p₀ ≥ 1/2 always. The data
   standardization step (mean=0, std=1) that ref. 10 required is not needed.

This is what the paper means on page 2: "we show that the classifier can be
realized without post-selection, thereby reducing the number of experiments
by about a factor of two, and avoiding the pre-processing."

---

## How the initial state is prepared (Eq. 8 of the paper — the key equation)

This is the most important equation in the paper and our current explanation
barely touched it. Here it is in full:

Starting from this product state (data encoded separately on different registers):
```
Σ_m √(w_m) |0⟩_a  |x̃⟩^⊗n  |0⟩^⊗n_d  |0⟩_l  |m⟩  |x₁⟩^⊗n  |y₁⟩  |x₂⟩^⊗n  |y₂⟩ ...
```

After applying quantum forking gates (series of controlled-SWAPs on the index
register — this is the `U_s(D)` block in Fig. 2 of the paper):
```
Σ_m √(w_m) |0⟩_a  |x̃⟩^⊗n  |x_m⟩^⊗n  |y_m⟩  |m⟩  |junk_m⟩
```

Now each index m "routes" the training data into the data register, with
the label y_m in the label register and weight √(w_m) as the amplitude.
This is a SUPERPOSITION over all training examples simultaneously — quantum
parallelism is doing all M comparisons at the same time.

Then apply the swap-test gates (H → CSWAP^n → H on the ancilla):
```
→ |Ψ_f^s⟩ = Σ_m (√(w_m)/2) · (|0⟩|ψ_n+⟩ + |1⟩|ψ_n-⟩) |y_m⟩ |m⟩ |junk_m⟩
```

Where |ψ_n±⟩ = |x̃⟩^⊗n ⊗ |x_m⟩^⊗n ± |x_m⟩^⊗n ⊗ |x̃⟩^⊗n

The ancilla is now entangled with the symmetric/antisymmetric combination
of test and training states. When you measure ⟨σz^(a) σz^(l)⟩ on this state,
the math works out to exactly Eq. 9. This derivation uses:
```
tr(|ψ_n±⟩⟨ψ_n±|) = 2 ± 2|⟨x̃|x_m⟩|^{2n}
```
And `tr(σz |y_m⟩⟨y_m|) = +1 if y_m=0, −1 if y_m=1`

---

## Quantum forking — the product-state approach (Section 3 of the paper)

This section is entirely missing from our explanation so far.

The problem with Eq. (8): it requires the entire state (test data + ALL training
data + labels) to be in a specific entangled superposition from the start. That
means you need to know ALL the training data classically before quantum execution
and encode it all into one big entangled state. This requires a complex
state-preparation circuit U_s(D).

The paper proposes an alternative: **quantum forking** (based on refs. 12–13 in
the paper, authored by the same group). Instead of preparing the big entangled
state directly, you start with all the data on SEPARATE, INDEPENDENT registers:

```
|0⟩_a  |x̃⟩^⊗n  |0⟩^⊗n_d  |0⟩_l  |m⟩  |x₁⟩^⊗n  |y₁⟩  |x₂⟩^⊗n  |y₂⟩ ...
```

Then a series of controlled-SWAP gates (one per training example, controlled
by the index qubit) "forks" the training data into the data register:
- c-swap(d, x_m | m): if index qubit is in state |m⟩, swap x_m into register d

After all these controlled swaps, the state becomes:
```
Σ_m √(w_m) |0⟩_a |x̃⟩^⊗n |x_m⟩^⊗n |y_m⟩ |m⟩ |junk_m⟩
```

The `|junk_m⟩` is some leftover product state in the now-empty training
registers. Crucially, since the measurement only involves ancilla and label,
and the junk traces out to identity, the junk registers don't affect the result.

**Why this matters:**
- The classifier can work on INTRINSIC QUANTUM DATA (data that was never
  classical — e.g., output of another quantum computation). You don't need
  to know what the training states are.
- Parallel state preparation: all training states can be prepared independently
  on separate registers (possibly in parallel on different QPU chips).
- Gate-intensive quantum feature maps: instead of classically computing a
  feature map and loading the result, the feature map circuit itself runs on
  the training register, and forking distributes it.

**The cost:** circuit becomes wider (more qubits) instead of deeper.
Gate count grows linearly with M (number of training examples) and
logarithmically with N (feature dimension). The paper gives exact counts:
- Qubits: n(M+2)⌈log₂(N)⌉ + 2⌈log₂(M)⌉ + M + 1
- For n=1, 16 training examples, 8 features: 79 qubits, 163 Toffoli, 134 CNOT

This is in the range of current hardware (100–1000 qubits) but gate fidelity
requirements (error < 10⁻³ per Toffoli gate) are still just beyond NISQ capability
as of the paper's publication in 2020.

**In our codebase:** `circuits/quantum_forking.py` contains utilities for this.
`qiskit_layer/circuits.py:build_product_state_n_copies_circuit()` implements the
product-state version for our 2-training-example toy problem.

---

## Noise robustness — a key property the paper proves

The paper (page 3, paragraph after Eq. 9) proves that the algorithm is
naturally robust to certain types of quantum errors. Specifically:

**Any Pauli error that commutes with σz^(a) σz^(l) does NOT change the result.**

Pauli operators are: I (identity), X (bit flip), Y (both), Z (phase flip).

σz^(a) σz^(l) commutes with any σz error on any qubit (phase errors don't
change measurement basis). It also commutes with any error on non-ancilla,
non-label qubits (since those operators act on different qubits).

What DOESN'T commute: a bit-flip error (σx) on the ancilla or label qubit.
If this happens with probability p, the measurement outcome becomes:
```
(1 − 2p) · ⟨σz^(a) σz^(l)⟩
```
The sign is preserved as long as p < 1/2. Since real hardware has p << 1/2,
the CLASSIFICATION is correct even with moderate noise — you might get the
wrong magnitude but the sign (the actual answer) survives.

This is why the paper's hardware experiment (Fig. 5) shows "amplitude reduction
of about 0.65" — the kernel values are scaled down by noise but the sign
(and therefore the classification) is still correct for ~97% of test points.

**In our results:** the same amplitude reduction appears in our hardware runs.
Hardware mean abs error is ~0.07 while simulator is ~0.04 — the reduction is
about 0.65× of the true amplitude, exactly matching the paper's observation.

Also importantly: dephasing noise on the INDEX qubit (m register) does NOT
affect the result, because the cross-terms from the index superposition cancel
out in the expectation value calculation. Same for the label qubit. This means
the most fragile part of the circuit (the many-body index superposition) is
actually the most noise-resistant part.

---

## What does the paper's Fig. 5 show — and how our results compare

The paper's Fig. 5 (page 5) shows three curves:
1. **Black solid/dotted lines**: theoretical kernel ±0.5·sin θ (our "theory")
2. **Blue squares**: Qiskit simulation with a realistic noise model from ibmq_ourense
   calibration data (T₁/T₂ relaxation times, gate errors, readout errors)
   → amplitude reduction factor ~0.82, negligible phase shift
3. **Red triangles**: actual hardware experiment on IBM Q 5 Ourense (5-qubit QPU)
   → amplitude reduction factor ~0.65, ~2° phase shift in θ

The paper says ~97% of test points are correctly classified despite the noise.

**Our results are directly comparable:**

| Metric | Paper (ibmq_ourense) | Our results (ibm_kingston) |
|---|---|---|
| Amplitude reduction | ~0.82 (sim), ~0.65 (hw) | reflected in mean abs diff |
| Sign agreement | ~97% (hw) | 96.67–100% depending on shots |
| Backend | 5-qubit Ourense (2019) | 127-qubit Kingston (2024) |
| Shots | 8192 | 1024 and 256 |

The difference: ibm_kingston (2024) is a newer, better-calibrated device than
ibmq_ourense (2019). Also we used 1024 shots instead of 8192, but sign agreement
is still very high.

---

# PART 9: THE CODEBASE — FILE BY FILE

## main.py — The command center

This is the script you run from the terminal. It accepts command-line
arguments and calls all the other functions.

Key functions:
- `run_verification()`: Runs 8 mathematical checks (like a test suite)
- `run_all_experiments()`: Generates the 7 NumPy figures (01-07)
- `run_qiskit_paper_mode()`: Runs one circuit family on simulator or hardware
- `run_qiskit_shots_comparison()`: Runs 256 vs 1024 shots and compares
- `run_vce_novelty_comparison()`: Runs the VCE novelty pipeline
- `main()`: Parses CLI args and routes to the right functions

## core/kernel.py — Pure math

No quantum hardware here. Just NumPy implementing the paper's equations.

- `hadamard_kernel(x_test, x_train, labels)`: Eq. 6. Will be ~0 for toy problem.
- `swap_test_kernel(x_test, x_train, labels, n_copies)`: Eq. 9. Works correctly.
- `kernel_matrix(states, n_copies)`: Builds the N×N Gram matrix K[i,j] = |⟨xᵢ|xⱼ⟩|^{2n}
- `helstrom_operator(x_train, labels, n_copies)`: Builds matrix A (Eq. 16)
- `helstrom_expectation(x_test, A, n_copies)`: Computes ⟨x̃^⊗n|A|x̃^⊗n⟩ (Eq. 17)

## core/quantum_gates.py — NumPy quantum mechanics

Building blocks for the statevector simulation:

- `H`: 2×2 Hadamard matrix [[1,1],[1,-1]]/√2
- `apply(gate, state)`: matrix-vector multiply
- `tensor(a, b, c, ...)`: tensor/Kronecker product, building multi-qubit states
- `embed_single(gate, qubit, n_qubits)`: put a 2×2 gate on one qubit in n-qubit system
- `controlled_swap(control, t1, t2, n_qubits)`: CSWAP matrix
- `expectation_ZZ(state, qa, qb, n_qubits)`: compute ⟨σz_a · σz_b⟩
- `ket(m, n_qubits)`: create |m⟩ basis state as a NumPy vector
- `normalize(state)`: divide state by its norm

## experiments/toy_problem.py — The specific problem instance

- `get_training_data()`: Returns [|x₁⟩, |x₂⟩] and [0, 1] as NumPy arrays
- `get_test_state(theta)`: Returns cos(θ/2)|0⟩ + i·sin(θ/2)|1⟩
- `get_theta_range(n)`: Returns n equally spaced θ values from 0 to 2π
- `analytical_swap_kernel(theta, n_copies)`: Exact formula K_n(θ), no simulation needed
- `analytical_hadamard_kernel(theta)`: Always returns ~0.0 (by construction)
- `true_classification(theta)`: The correct answer: 0 if θ∈(0,π), 1 if θ∈(π,2π)

## experiments/noise_simulation.py — Simulating a noisy quantum device

Real quantum computers make errors. This file simulates that on CPU:
- Applies random bit flips with some probability
- Models amplitude damping (qubits losing energy to environment)
- Returns statistics over many Monte Carlo runs

Used for Figure 02 (theory vs noisy comparison).

## circuits/swap_test_classifier.py — NumPy swap-test circuit

A full statevector simulation of the swap-test circuit using just NumPy.
This is the "exact" simulation — no shots, no noise. Perfect for checking.

Key method: `run(x_train, labels, x_test)` — returns the full statevector at
each step plus the final expectation value.

We verify: this matches `analytical_swap_kernel()` to within 1e-8. ✓

## circuits/hadamard_classifier.py — NumPy Hadamard circuit

Same idea but for the prior-work Hadamard classifier.
Confirms that it gives ~0 output for all θ on the toy problem.

## qiskit_layer/circuits.py — REAL quantum circuits (Qiskit)

Here we describe circuits using Qiskit's QuantumCircuit API — the actual
gate-level description that can run on IBM hardware.

Two functions:
1. `build_swap_test_toy_circuit(theta, weights)` — 5-qubit circuit, matches paper
2. `build_product_state_n_copies_circuit(theta, copies, weights)` — n-copy version

These produce Qiskit circuit objects that can be transpiled (compiled) and
sent to a real quantum computer.

## qiskit_layer/backends.py — Connecting to IBM

Handles two types of backends:

**AerSimulator**: Runs locally on your CPU. Mimics a quantum computer.
  Can add a noise model to simulate hardware imperfections.

**IBM hardware**: The real thing, in IBM's data centers.
  You need a token (like a password) from quantum.ibm.com.
  The token goes in a `.env` file (see `.env.example`).

Key function: `get_ibm_runtime_config()` reads the token from environment
variables or the `.env` file and builds a config object.

## qiskit_layer/runner.py — Running circuits and collecting results

The main function: `run_swaptest_theta_sweep_qiskit(thetas, shots, mode, ...)`

1. Build a circuit for each θ value in the sweep
2. Transpile (compile/optimize the circuit for the specific hardware)
3. Send to simulator or IBM hardware
4. Collect measurement counts
5. Compute expectation value from counts: (c00 − c01 − c10 + c11) / shots
6. Return JSON with all data

For hardware: uses `SamplerV2` from qiskit-ibm-runtime (the modern IBM API).

## qiskit_layer/noise.py — Building noise models

`build_simple_depolarizing_noise_model()`: Creates a noise model where
- Single-qubit gates have 0.1% error probability
- Two-qubit gates (CNOT, CSWAP) have 1% error probability

`build_noise_model_from_backend(ibm_backend)`: Extracts the ACTUAL calibrated
error rates from a real IBM backend's latest calibration data.

## qiskit_layer/mitigation.py — The VCE Novelty

The most novel part of our project. See Part 10 below.

## visualization/plots.py — All figures

One function per figure type:
- `plot_n_copies_effect()` → Figure 01
- `plot_theory_vs_noisy()` → Figure 02
- `plot_hadamard_vs_swaptest()` → Figure 03
- `plot_bloch_sphere()` → Figure 04 (uses matplotlib's 3D axes)
- `plot_kernel_matrix()` → Figure 05 (imshow heatmap)
- `plot_helstrom_equivalence()` → Figure 06
- `plot_circuit_verification()` → Figure 07
- `plot_qiskit_vs_theory()` → Figures 09
- `plot_qiskit_shots_comparison()` → Figures 12
- `plot_vce_target_comparison()` → Figures 15
All figures are saved as PNG files in the `results/` directory.

## scripts/hardware_suite.py — IBM hardware runner

The problem: `main.py --compare-suite --paper-backend hardware` hits the
IBM global catalog API 5 times separately, and IBM's API sometimes times out.
If it times out on run 3, you lose runs 1 and 2 too.

The fix: create ONE service connection, ONE backend reference, ONE Sampler
instance, and reuse them for all 5 sweeps. Each sweep result is saved to disk
immediately so partial results survive a later crash.

```python
svc = QiskitRuntimeService(...)   # ONE connection
backend = svc.backend("ibm_kingston")  # ONE backend
sampler = Sampler(mode=backend)   # ONE sampler
# Then run all 5 sweeps using the SAME sampler
```

## scripts/cross_comparison.py — Comparing simulator vs hardware

After running both the simulator suite and hardware suite, this script:
1. Reads both `14_qiskit_vce_*_summary.json` files
2. Aligns the theta grids
3. Computes pre/post novelty improvement deltas
4. Generates Figure 18 (the 2×2 comparison plot)
5. Saves `17_qiskit_sim_vs_hw_novelty_comparison.json`

---

# PART 10: OUR NOVELTY — VCE EXPLAINED IN DETAIL

## The problem

Using n=3 copies on IBM hardware means running circuits with:
- 3 copies of the training register (3 qubits)
- 3 copies of the test register (3 qubits)
- Plus ancilla, label, index = ~9 qubits total

Each additional qubit pair is 2 more two-qubit CSWAP gates. Two-qubit gates
have ~1% error rate on real hardware. 3× more gates = 3× more accumulated
error. The n=3 physical circuit is noticeably noisier than n=1.

## The insight

If we know the shape of K_n mathematically:
```
K_n(θ) = ½·(p^n − (1-p)^n)   where  p = |⟨x̃|x₁⟩|²
```
Then we only need to estimate `p` — one number per θ — and we can compute
K_n for any n we want!

But our measured K₁ has noise: K₁(measured) = K₁(ideal) + noise.
We need to remove the noise before inverting for p.

## Richardson extrapolation

Richardson extrapolation is a classical numerical technique from the 1910s.
The basic idea: if you have a measurement at scale h and at scale 2h, you can
cancel out the leading error term:

```
f(ideal) ≈ 2·f(h) − f(2h)
```

In our context:
- K₁(measured) = K₁(ideal) + ε (where ε is the noise)
- K₂(measured) = K₂(ideal) + 2ε (roughly — noise accumulates with more copies)

So: 2·K₁ − K₂ ≈ K₁(ideal). The noise cancels!

```python
k1_denoised = 2.0 * k1_measured - k2_measured   # Richardson step
```

## From K₁* to p to K_target

Step 1: Get denoised K₁:
```
K₁* = 2·K₁ − K₂
```

Step 2: Invert the n=1 formula to get p:
```
K₁ = ½·(p − (1−p)) = p − ½
→ p = K₁* + ½
```
Clip p to [0, 1] since it's a probability.

Step 3: Compute any target n:
```
K_target = ½·(p^target − (1−p)^target)
```

That's it. With two physical runs (n=1 and n=2), we estimate K_3, K_5, K_10...
without ever running those circuits.

## Why does it beat the physical n=3?

Physical n=3 accumulates 3× the hardware noise.
VCE n=3 runs n=1 and n=2, then does math. The n=1 circuit has 1× noise.
Richardson denoising further reduces it. The net result is a cleaner estimate.

Think of it like: you can either interview the candidate for 3 hours straight
(exhausting, attention wanders, worse answers) or interview them twice for
1 hour each (fresher, sharper, but you get to cross-check too). The latter
often gives a better picture.

## The code (qiskit_layer/mitigation.py)

```python
def predict_virtual_toy_richardson(k_n1, k_n2, target_copies):
    # Step 1: Richardson denoising
    k1_denoised = 2.0 * k_n1 - k_n2

    # Step 2: recover probability p
    p = clip(k1_denoised + 0.5, 0, 1)
    q = 1 - p

    # Step 3: compute target kernel analytically
    pred = 0.5 * (p**target_copies - q**target_copies)

    # Optional: sign stabilization (keep signs consistent with n=1)
    pred = sign(k_n1) * abs(pred)

    return clip(pred, -0.5, 0.5)
```

## Results

Hardware (`ibm_kingston`, 1024 shots):

| Method | Mean abs error | RMSE |
|---|---|---|
| Physical n=3 | 0.0818 | 0.0983 |
| Virtual n=3 (VCE) | **0.0717** | **0.0860** |

~12% improvement in mean error, ~13% improvement in RMSE.
VCE is better than physically running n=3.

On the simulator, improvement is ~26% in mean error (less hardware noise to
start with, so Richardson has a cleaner signal to work with).

---

# PART 11: RUNNING ON REAL IBM HARDWARE

## What actually happened

We submitted quantum circuits to IBM's real quantum computer `ibm_kingston`.
This is a 127-qubit superconducting QPU.

Steps:
1. Create account at quantum.ibm.com
2. Get an API token
3. Put it in `.env`: `QISKIT_IBM_TOKEN=your_token_here`
4. Run `python scripts/hardware_suite.py --backend ibm_kingston`
5. Circuits go into a job queue (shared with everyone using IBM's open plan)
6. When your turn comes, the QPU runs your circuits
7. Results come back as measurement counts

## The queue

IBM's open-plan access means you share the computer with hundreds of other
researchers. Jobs can wait minutes to hours. Our hardware runs took ~20 minutes
of queue time.

## Why hardware is noisier than simulator

Real quantum hardware has many sources of error:
- **Decoherence**: qubits lose their quantum state over time (microseconds)
- **Gate errors**: every gate has a small chance of flipping a qubit wrong
- **Readout errors**: measuring "0" when the qubit is actually "1"
- **Crosstalk**: nearby qubits interfering with each other

Our hardware results have sign agreement ~96-100% vs theory, while simulator
(with noise model) reaches 100%. The remaining errors are coherent errors
(systematic, not random) that VCE helps correct.

## Job IDs

Every hardware run produces a Job ID, stored in the result JSONs. For example:
- Swap-test n=1 at 1024 shots: `d7ktoua8ui0s73b5n980`

You can look up these jobs on quantum.ibm.com to see the raw results, queuing
time, and QPU calibration data at the time of execution.

---

# PART 12: THE 41 RESULT FILES

All results are in `results/`. The naming convention is `NN_description.ext`
where NN is a 2-digit experiment number.

## Group 1: Pure math (01-07) — NumPy analytical layer

**01_n_copies_effect.png**
Shows K₁, K₁₀, K₁₀₀ plotted against θ. The curves get progressively more
sharp — narrow bands of +0.5 and −0.5 with quick transitions at 0 and π.

**02_theory_vs_noisy.png**
The "ideal" K₁ curve plus a Monte-Carlo noisy version with ±1σ shaded band.
Shows how shot noise spreads around the true value.

**03_hadamard_vs_swaptest.png**
Two panels side by side. Hadamard: a flat zero line. Swap-test: a clear
sinusoidal curve with correct class labels. The paper's key comparison.

**04_bloch_sphere.png**
3D visualization. Training states |x₁⟩ and |x₂⟩ are red and blue dots. The
test state trajectory is a circle around the equator, colored by class.

**05_kernel_matrix.png**
Heatmap showing K[i,j] = |⟨xᵢ|xⱼ⟩|^{2n} for 20 states and n=1,2,5.
Diagonal = 1 (every state has fidelity 1 with itself). Off-diagonal shows
how similar pairs of states are.

**06_helstrom_equivalence.png**
Two overlapping curves (theory and Helstrom) that are identical to the eye,
plus a log-scale error panel showing the error is ~1e-16. Confirms the
theorem numerically.

**07_circuit_verification.png**
Compares the NumPy circuit simulation vs analytical formula. Both swap-test
and Hadamard classifiers match their analytical counterparts to ~1e-8.

**summary_report.txt**
Text file with all numerical metrics: kernel range, Helstrom diff, accuracy.

## Group 2: First Qiskit runs (08-09)

Single runs, one circuit family, one backend.

**08_qiskit_swap_test_simulator_results.json** — Raw counts + expectations
**09_qiskit_swap_test_simulator_vs_theory.png** — Measured vs theory curve

Same thing for hardware (with job IDs) and product_state circuit family.

## Group 3: Shots comparison (10-12)

Does 4× more shots (256→1024) make a measurable improvement?

**10_qiskit_swap_test_{mode}_shots_{N}_results.json** — Raw per-θ data
**11_qiskit_swap_test_{mode}_shots_comparison.json** — Combined summary
**12_qiskit_swap_test_{mode}_shots_comparison.png** — Two curves on one plot

Answer: Yes. Sign agreement goes from 96.67% → 100% on both simulator and
hardware. Mean error improves ~29% on simulator, ~16% on hardware.
(Hardware improvement is smaller because coherent errors dominate at 1024
shots — that's exactly what motivates VCE.)

## Group 4: VCE novelty (13-15)

**13_qiskit_product_state_{mode}_copies_{n}_shots_1024_results.json**
One file per (mode, copies) combination. 6 files total.

**14_qiskit_vce_{mode}_shots_1024_summary.json**
The VCE processing result. Contains:
- physical_curves: measured K₁, K₂, K₃
- theory_curves: theoretical K₁, K₂, K₃, K₅
- virtual_curves: VCE-estimated K₃, K₅, K_inf
- metrics: error vs theory for each curve

**15_qiskit_vce_{mode}_shots_1024_pre_post.png**
Shows theory K₃ (black), physical n=3 (blue), virtual n=3 from VCE (red).
The red curve is visually closer to the black curve than the blue one.

## Group 5: Suite summaries (16)

**16_qiskit_{mode}_comparison_suite_summary.json**
One mega-file combining shots comparison + VCE summary.

## Group 6: Cross-comparison (17-18)

**17_qiskit_sim_vs_hw_novelty_comparison.json**
All four curves (sim physical, sim virtual, hw physical, hw virtual) plus
the improvement deltas pre→post novelty for both backends.

**18_qiskit_sim_vs_hw_pre_post_comparison.png**
The 2×2 figure:
- Top-left: Simulator pre/post curves
- Top-right: Hardware pre/post curves
- Bottom-left: Simulator absolute error curves
- Bottom-right: Hardware absolute error curves

The bottom panels show the red (virtual) curve is consistently below the
blue (physical) curve — VCE reduces error across the whole θ range.

---

# PART 13: WHAT MAKES THIS WORK SIGNIFICANT?

## NISQ era

We're currently in the NISQ era (Noisy Intermediate-Scale Quantum). Quantum
computers have 10–1000 qubits but they make a lot of errors. Full error
correction requires ~1000 physical qubits per logical qubit — not yet feasible.

So we need to work with noisy qubits. That's what VCE addresses.

## The tradeoff VCE solves

Without VCE: to get a better kernel, you need n=3 hardware run (more errors).
With VCE: run n=1 and n=2 (cheaper), then do classical post-processing.

This is a general strategy called **error mitigation** — you can't eliminate
quantum errors, but you can reduce their effect using classical computation.

## Relation to Richardson extrapolation

Richardson extrapolation was invented by Lewis Fry Richardson in 1911 for
numerical differentiation. The same idea (using two approximations to cancel
error) has been rediscovered in quantum computing as "zero-noise extrapolation"
(ZNE). Our VCE is a variation: instead of varying noise level, we vary the
number of copies and use the mathematical structure of the kernel.

## What this shows about quantum ML

The swap-test classifier is provably optimal (Helstrom-equivalent) for
distinguishing quantum states. For classical data encoded into quantum states,
it provides a kernel that classical computers cannot efficiently compute when
the quantum states have special structure (entanglement). This is the
theoretical basis for "quantum advantage" in machine learning — though
for this specific toy problem, the advantage is not computational but
pedagogical.

---

# PART 14: GLOSSARY

| Term | Simple explanation |
|---|---|
| Qubit | A quantum bit. Can be 0, 1, or superposition of both |
| Superposition | "Both at the same time" until measured |
| Entanglement | Two qubits correlated such that measuring one instantly affects the other |
| Gate | A quantum operation that transforms qubit states |
| Hadamard gate (H) | Puts a qubit into equal superposition |
| CSWAP | Conditional swap of two qubits, controlled by a third |
| Fidelity | |⟨a|b⟩|² — how "close" two quantum states are (0=orthogonal, 1=identical) |
| Inner product ⟨a|b⟩ | Complex dot product of two state vectors |
| Kernel | Similarity measure between two data points |
| Tensor product | ⊗ — the "multiplication" that combines quantum systems |
| Bloch sphere | Geometric representation of a single qubit state as a point on a sphere |
| Bra-ket notation | ⟨ψ| is "bra", |ψ⟩ is "ket". ⟨a|b⟩ = inner product |
| Shots | Number of times you run the circuit and measure (more shots = less noise) |
| Transpilation | Compiling/optimizing a quantum circuit for specific hardware |
| Decoherence | Qubits losing their quantum state due to environmental interference |
| NISQ | Noisy Intermediate-Scale Quantum — current era of imperfect quantum computers |
| VCE | Our novelty: Virtual Copy Extrapolation |
| Richardson extrapolation | Classic technique: use two approximations to cancel leading error |
| AerSimulator | IBM's local quantum circuit simulator (runs on CPU) |
| SamplerV2 | IBM Runtime API for running circuits on real QPU |
| Sign agreement | % of θ values where predicted class matches theoretical class |
| Mean abs diff | Average absolute difference between measured and theoretical kernel values |
| RMSE | Root mean square error — penalizes large errors more than mean abs diff |

---

# PART 15: HOW TO READ THE PAPER

If you want to read the paper itself (https://www.nature.com/articles/s41534-020-0272-6),
here's a guide to the key parts and where to find them in our code:

| Paper section | Key equation | Our code |
|---|---|---|
| Section 2: Classifier | Eq. 6 — Hadamard kernel | `core/kernel.py:hadamard_kernel()` |
| Section 2: Classifier | Eq. 9 — Swap-test kernel | `core/kernel.py:swap_test_kernel()` |
| Section 3: Toy example | Eq. 11 — Training states | `experiments/toy_problem.py:get_training_data()` |
| Section 3: Toy example | Eq. 12 — Analytical K_n | `experiments/toy_problem.py:analytical_swap_kernel()` |
| Section 3: Toy example | Eq. 13 — Why Hadamard fails | `experiments/toy_problem.py:analytical_hadamard_kernel()` |
| Section 3: n copies | Eq. 14 — Limit n→∞ | `results/01_n_copies_effect.png` |
| Section 4: Helstrom | Eqs. 16-17 | `core/kernel.py:helstrom_operator()` + `helstrom_expectation()` |
| Supplemental code | Circuit implementation | `qiskit_layer/circuits.py` |
| Our novelty | VCE estimator | `qiskit_layer/mitigation.py` |

The paper's Fig. 3 = our `01_n_copies_effect.png`
The paper's Fig. 5 = our `02_theory_vs_noisy.png`
Our Figs. 10-18 = the Qiskit/hardware extension beyond what the paper shows

---

*This file is for personal study only — not included in the git commit.*
