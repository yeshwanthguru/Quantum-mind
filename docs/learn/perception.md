# Perception and computer vision

Perception turns raw sensor signals into things a robot can reason about: objects, people, places and
words. **Computer vision** does this for images. Every perception output is uncertain, so a robot
needs not just an answer ("cup") but a trustworthy **confidence**, and it often has to combine several
uncertain sources (camera, speech, gaze).

```{mermaid}
flowchart LR
    C["camera"]:::in --> DET["object detector<br/>labels + scores"]:::op
    M["microphone"]:::in --> ASR["speech recogniser<br/>n-best list"]:::op
    G["gaze / pointing"]:::in --> DIR["direction estimate"]:::op
    DET --> CAL["calibrate<br/>scores → probabilities"]:::op
    ASR --> CAL
    DIR --> CAL
    CAL --> F["fuse<br/>Bayes · Dempster-Shafer · quantum-like"]:::op --> B["belief over targets"]:::out
    classDef in fill:#0b2a4a,stroke:#79c0ff,color:#e6edf3
    classDef op fill:#2a1b3d,stroke:#d2a8ff,color:#e6edf3
    classDef out fill:#0f2e1a,stroke:#7ee787,color:#e6edf3
```

## From the basics

Image formation
: A camera projects 3D points onto a 2D image (the pinhole model). Depth cameras and lidar measure
  distance directly; stereo vision recovers it from two views.

Classification, detection and segmentation
: **Classification** says what is in an image; **detection** draws boxes around objects with scores;
  **segmentation** labels every pixel. Modern systems use deep networks (CNNs, vision transformers).

Confidence and calibration
: A detector's score is not a probability until it is calibrated. Calibrated confidences let a robot
  decide when to trust a detection and when to ask.

Sensor fusion
: Combining sources. **Bayesian fusion** multiplies likelihoods, assuming the sources are independent
  given the true target. **Dempster-Shafer** combines belief masses and can express ignorance.
  **Quantum-like fusion** represents each source as a basis in a shared vector space, so sources can
  be incompatible and their order can matter.

Multistable perception
: Some images (the Necker cube) have two interpretations and perception flips between them. The
  **dwell time** in each interpretation has a characteristic distribution; the quantum Zeno model
  predicts how it scales with how often the percept is "checked".

Vision-language models
: Networks that connect images with text, used to ground requests such as "the red cup on the left" in
  what the camera sees.

## A small example: Bayesian fusion of two sources

The detector gives target probabilities (0.6, 0.3, 0.1) for three cups; speech gives (0.2, 0.7, 0.1).

```python
import numpy as np
from quantum_mind.applications.fusion import bayes_fusion
det = np.array([0.6, 0.3, 0.1])
asr = np.array([0.2, 0.7, 0.1])
print(bayes_fusion([det, asr]).round(3))     # the two sources together favour cup 2
```

## Where Quantum Mind fits

The library recalibrates perception outputs, fuses them with three methods that can be compared on
held-out data, and models bistable perception with a quantum Zeno model against Markov and
gamma-renewal baselines. On simulated data the quantum-like fusion had the lowest held-out log loss
(0.740 against 0.755 for Bayes), with weakly identifiable angles.

- [Tutorial: multimodal fusion](../tutorials/08_fusion.ipynb)
- [Tutorial: bistable perception](../tutorials/09_bistable_perception.ipynb)
- [Model atlas: perception](../atlas/perception.md)

## References

- Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer.
- Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge
  University Press.
- Ernst, M. O., & Banks, M. S. (2002). Humans integrate visual and haptic information in a
  statistically optimal fashion. *Nature*, 415, 429-433.
- Shafer, G. (1976). *A Mathematical Theory of Evidence*. Princeton University Press.
- Atmanspacher, H., Filk, T., & Römer, H. (2004). Quantum Zeno features of bistable perception.
  *Biological Cybernetics*, 90, 33-40.
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision.
  *Proceedings of ICML*, 8748-8763.
