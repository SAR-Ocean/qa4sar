# `rvlHeading` — Angle Convention, Projection Formula, and Incidence Angle

This note documents the geometric reasoning behind `_project_currents_to_radial`
in `sar_validation/core/collocation.py` and addresses two questions that arise
when reading the code:

1. Where does the `- 90` in `heading_rad = np.radians(heading_deg - 90.0)` come from?
2. Should the SAR incidence angle also be included in the projection?

---

## Background

The Sentinel-1 L2 OCN RVL product provides `rvlRadVel` — the radial (line-of-sight)
surface velocity in m/s — alongside the geometry metadata variable `rvlHeading`.
To compare in-situ ocean current vectors (`EWCT`, `NSCT`) against this scalar
SAR measurement, the current vector must be projected onto the same direction that
the radar is looking. The result is stored as `rvlRadVel_projection`.

The projection is computed by `_project_currents_to_radial`:

```python
heading_rad = np.radians(heading_deg - 90.0)
return ewct * np.cos(heading_rad) + nsct * np.sin(heading_rad)
```

---

## 1. The `rvlHeading` angle convention

`rvlHeading` is the satellite's **along-track (azimuth) direction** stored in
**mathematical angle convention** — counter-clockwise from East, where:

| Direction | `rvlHeading` |
|-----------|-------------|
| East      | 0°          |
| North     | 90°         |
| West      | 180°        |
| South     | 270°        |

This is *not* the standard navigation/compass bearing (clockwise from North).
The distinction matters when reading the `- 90` in the formula.

**Evidence from the test suite** (`tests/test_collocation.py`, line 750):

```python
"rvlHeading": np.array([90.0]),  # heading_rad = radians(90-90)=0
# heading 90 -> heading_rad 0 -> projection = EWCT*cos0 + NSCT*sin0 = EWCT
assert proj == pytest.approx(0.4, abs=1e-6)  # EWCT = 0.4
```

`rvlHeading = 90°` produces a pure eastward projection (only `EWCT` contributes).
In math convention, 90° = North, so this encodes a **northbound satellite** whose
range direction is East — correct for a right-looking sensor.

---

## 2. Why `- 90°` gives the range direction

For a **right-looking** SAR (standard Sentinel-1 configuration), the radar looks
90° **clockwise** from the flight direction. In math convention (where
counter-clockwise is positive), a 90° clockwise rotation is expressed as
**subtracting 90°**:

$$\alpha_\text{range} = \text{rvlHeading} - 90°$$

The horizontal unit vector pointing in the range direction (in East–North
coordinates) is:

$$\hat{r} = \bigl(\cos\alpha_\text{range},\; \sin\alpha_\text{range}\bigr)$$

The horizontal projection of the current onto this direction is the dot product:

$$v_\text{projection} = \text{EWCT} \cdot \cos(\text{rvlHeading} - 90°)
                      + \text{NSCT} \cdot \sin(\text{rvlHeading} - 90°)$$

which is exactly what the code computes.

### Worked examples

| Orbit | `rvlHeading` | `rvlHeading − 90°` | Range direction | Physical check |
|---|---|---|---|---|
| Ascending (northbound) | ≈ 90° | ≈ 0° | East | Right-looking northbound → looks East ✓ |
| Descending (southbound) | ≈ 270° | ≈ 180° | West | Right-looking southbound → looks West ✓ |

---

## 3. Does the orbit direction change the sign?

**No.** The same `- 90°` applies to both ascending and descending passes.

The orbit direction is already encoded in `rvlHeading` itself:
- Ascending: rvlHeading ≈ 90° (northward) → range = 0° (East)
- Descending: rvlHeading ≈ 270° (southward) → range = 180° (West)

For a descending pass looking West, an eastward current moves *away* from the
satellite, so the radial velocity should be negative. The formula gives:

$$\cos(180°) = -1 \implies v_\text{projection} = -\text{EWCT}$$

which has the correct sign. No code change is needed for descending passes.

You would only need `+ 90°` if the SAR were **left-looking**, which is not the
standard Sentinel-1 acquisition mode.

---

## 4. Should the incidence angle be included?

**No — and this is by design.**

The naive formula for the 3-D slant-range velocity is:

$$v_\text{slant} = \sin\theta \cdot v_\text{horizontal range}$$

where $\theta$ is the local incidence angle. However, the Sentinel-1 L2 processor
*already divides by $\sin\theta$* when computing `rvlRadVel`, using:

$$\text{rvlRadVel} = \frac{-f_{DC} \cdot \lambda}{2 \cdot \sin\theta_\text{local}}$$

The result is the **horizontal range velocity** (units: m/s), not the raw
slant-range Doppler rate. Both sides of the comparison are therefore in the same
physical units and on the same geometric reference plane — no additional
$\sin\theta$ factor is needed in the projection formula.

This is why `rvlIncidenceAngle` is extracted and stored in the DataTree (see
`datatree_converter.py`) but is not used in collocation or statistics: it is
retained as geometry metadata for traceability and potential future use, but the
L2 product has already applied it.

---

## 5. Summary

| Question | Answer |
|---|---|
| What convention does `rvlHeading` use? | Math convention — counter-clockwise from East (North = 90°) |
| Why `- 90°`? | Converts along-track to range direction: 90° CW = −90° in math convention (right-looking SAR) |
| Same formula for descending passes? | Yes — `rvlHeading` ≈ 270° for southbound, so range = 180° (West), giving the correct sign automatically |
| Is incidence angle needed? | No — `rvlRadVel` is already the horizontal range velocity (L2 processor divided by sin θ) |

---

## 6. Caveats

The `rvlHeading` angle convention (math, CCW from East) is inferred from geometric
consistency and the test suite rather than directly verified against the ESA Level-2
product format specification document, which was not directly accessible at the time
of writing. The ESA algorithm definition document should be consulted to confirm the
convention if the formula is ever challenged by a comparison that shows a systematic
sign error on one orbit direction but not the other.

The reference given in the docstring — Martin, Gommenginger, Jacob & Staneva (2022),
RSE 268:112758 — describes the overall SAR current validation methodology; the
specific trigonometric form of the projection formula follows from first principles
as shown above.
