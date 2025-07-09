?? Probabilistic Matching Model

This tracker moves beyond greedy nearest-neighbor matching and instead uses a probabilistic model to determine the most likely object associations across frames.
Matching Probability

The likelihood that a detection j in frame t belongs to an existing track i from frame t - k is modeled as:

P(match) = Pg(k) × Pd(distance) × Ps(size_ratio)

We then use -log(P(match)) as a cost in a global assignment problem (e.g. Hungarian or A* search).
Components:
? Pg(k): Probability of Gap

Models the chance an object goes undetected for k frames:

    Pg(1) = 0.8 ? 80% chance we recover a missed detection after 1 frame

    Pg(k) = Pg(1)^k for k ? 3, and 0 for k > 3

This encodes a soft limit on track memory length.
? Pd(d): Probability of Match by Displacement

Models how likely two detections are the same based on their motion:

Pd(d) = Pd(1) ^ (d / max(h, w))

    Pd(1) = 0.3 ? 30% chance if offset is 1× object size

    Exponentially decays as normalized distance increases

This allows flexible matching for small motions, but punishes big jumps.
? Ps(S1/S2): Probability of Match by Size Ratio

Models how likely two detections are the same based on their area ratio:

Ps(r) = Ps(1) ^ |log?(r)|

    Ps(1) = 0.5 ? 50% chance if area is identical

    Ps(2) = Ps(1/2) = 0.25 ? area doubled or halved

    Drop-off is symmetric and logarithmic

This captures the fact that objects may change size slightly, but dramatic size changes are rare unless it's a new object.

This approach allows the system to:

    Model uncertainty in detection

    Prefer consistent assignments across frames

    Handle occlusion, reappearance, and similar object ambiguity more robustly than simple greedy strategies