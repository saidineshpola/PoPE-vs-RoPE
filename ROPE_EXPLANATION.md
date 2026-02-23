# Understanding RoPE: The Rotation Formula

## The Core Formula

```
rotated_vector = cos(θ) * vector + rotate_half(vector) * sin(θ)
```

This formula applies a **2D rotation** to pairs of dimensions in your embedding vector.

## Breaking It Down

### 1. What is `rotate_half(x)`?

The `rotate_half` function splits a vector into pairs and rotates each pair by 90 degrees:

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)  # Split into two halves
    return torch.cat([-x2, x1], dim=-1)  # Swap and negate: [x1, x2] -> [-x2, x1]
```

**Example:**
- Input: `[a, b, c, d, e, f]` (6 dimensions)
- Split: `[a, b]`, `[c, d]`, `[e, f]` (3 pairs)
- Output: `[-b, a, -d, c, -f, e]`

This is equivalent to multiplying by the matrix:
```
[0  -1]
[1   0]
```
which rotates a 2D vector by 90 degrees counterclockwise.

### 2. The 2D Rotation Matrix

In 2D, rotating a vector `[x, y]` by angle `θ` uses this matrix:

```
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]
```

So:
```
[x']   [cos(θ)  -sin(θ)] [x]
[y'] = [sin(θ)   cos(θ)] [y]
```

Which expands to:
```
x' = x * cos(θ) - y * sin(θ)
y' = x * sin(θ) + y * cos(θ)
```

### 3. How RoPE Implements This

RoPE uses a clever trick to compute this rotation efficiently:

```
x' = x * cos(θ) + rotate_half([x, y])[0] * sin(θ)
y' = y * cos(θ) + rotate_half([x, y])[1] * sin(θ)
```

Since `rotate_half([x, y]) = [-y, x]`:
- `rotate_half([x, y])[0] = -y`
- `rotate_half([x, y])[1] = x`

So:
```
x' = x * cos(θ) + (-y) * sin(θ) = x * cos(θ) - y * sin(θ)  ✓
y' = y * cos(θ) + x * sin(θ) = x * sin(θ) + y * cos(θ)    ✓
```

This matches the standard rotation matrix!

### 4. Why This Works for Position Encoding

For each position `pos` and dimension pair `i`, RoPE computes:
```
θ_i(pos) = pos * (10000^(-2i/d))
```

Where:
- `pos` = position in sequence (0, 1, 2, ...)
- `i` = which dimension pair (0, 1, 2, ...)
- `d` = total head dimension
- `10000` = base frequency

**Key insight:** The rotation angle depends on:
1. **Position** - different positions rotate by different amounts
2. **Dimension pair** - different pairs rotate at different frequencies

### 5. Visual Example

Imagine a 2D vector `[x, y]` at position 0:
- Position 0: `θ = 0`, so `cos(0) = 1`, `sin(0) = 0`
  - Result: `[x, y]` (no rotation)

- Position 1: `θ = θ₁`, so `cos(θ₁) = c`, `sin(θ₁) = s`
  - Result: `[x*c - y*s, x*s + y*c]` (rotated by θ₁)

- Position 2: `θ = 2*θ₁`, so `cos(2θ₁) = c₂`, `sin(2θ₁) = s₂`
  - Result: `[x*c₂ - y*s₂, x*s₂ + y*c₂]` (rotated by 2θ₁)

### 6. Why This Encodes Relative Position

When computing attention between position `i` and `j`:
- Query at position `i`: rotated by `θ_i`
- Key at position `j`: rotated by `θ_j`

The attention score becomes:
```
Q_i · K_j = (R(θ_i) * q) · (R(θ_j) * k)
```

Using rotation properties:
```
= q · R(θ_j - θ_i) · k
```

The attention depends on **θ_j - θ_i**, which is the **relative position** (j - i)!

This is why RoPE naturally encodes relative positions - the rotation difference between positions encodes their distance.

## Summary

- `cos(θ) * x`: Scales the original vector by cosine
- `rotate_half(x) * sin(θ)`: Adds a 90-degree rotated version scaled by sine
- Together: They implement a 2D rotation matrix efficiently
- The rotation angle `θ` depends on position, encoding positional information
- Different dimension pairs rotate at different frequencies, creating a rich encoding
