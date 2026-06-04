## Assignment Information
Assignment: Control dataset [individual] (15% of the grade)
Students will submit a blog post with a motivation of why their control dataset precisely tests a particular property, where the property is motivated with a link to one or multiple relevant papers. The blog should include a description of the control dataset with some examples, and how the dataset was generated; this should include links to the dataset and the code that was used to generate the dataset (e.g. on github/gitlab). 

Rubric: Penalties for the Control-dataset (blog):

- Not enough effort shown.
- Motivation is not sufficiently explained.
- No link to ML/DL research paper(s)
- Not clear how the control-dataset matches the motivation.
- No examples shown.
- Not explained how the data is generated.

Rubric: Penalties for all Writing (Storyline and blogs):

- Using a term before defining/motivating it.
- Too much unnecessary detail: words can be removed without significantly changing the storyline. Each word should have a reason to be there. 
- Unclear logical reasoning step.
- Inconsistent use of terminology. Use a single term for a single concept; 1-to-1.
- Writing too verbose / full sentences: bullet point should be one-two lines, one sentence, grammar optional
- Too many topics per storyline bullet point 
- Too many storyline bullet points. 
- The text is not stand-alone; it's not peer understandable.


### The JPEG re-encoding confound dataset

**Property and motivation.** ImageNet-C contains an admitted, unquantified confound. On page 3 the paper states that ImageNet-C images are saved as lightly compressed JPEGs, so an image corrupted by Gaussian noise is also slightly corrupted by JPEG compression. The benchmark therefore never measures the corruption _c_ in isolation; it measures _JPEG(c(x))_. The property your dataset would test is precisely this: how much of each corruption's reported Corruption Error is attributable to the named corruption versus the JPEG save step, and does this leakage differ across corruption types? This maps directly onto experimental question 5a in your storyline ("is the score fair and unbiased?").

**The control structure.** Take N clean images stored losslessly. For every corruption _c_ and severity _s_, generate matched variants that differ in exactly one thing, the save format: corruption applied and saved as lossless PNG, versus corruption applied and saved as JPEG at the quality the Hendrycks pipeline uses. Add a "clean image, JPEG-saved" arm to isolate JPEG acting alone. The control arm is the clean lossless image. The single isolated variable is the JPEG re-encode. Everything else (image, corruption, severity, random seed) is held identical, which is what makes a measured error gap attributable to JPEG and nothing else.

**Generation.** Reuse the original corruption functions so there is no ambiguity about what "the corruption" is: the imagecorruptions package is pip-installable and exposes a corrupt function taking a corruption name and severity, and it packages Hendrycks's original code. The only logic you add is `Image.save(format='PNG')` versus `Image.save(format='JPEG', quality=q)`. Inspect `make_imagenet_c.py` in github.com/hendrycks/robustness to find the exact quality setting, and sweep a couple of quality values (for example 85 and 95) so the result is not tied to one guess. [GitHub](https://github.com/bethgelab/imagecorruptions)

**Examples to show.** Clean PNG, noise-corrupted PNG, noise-corrupted JPEG, and the pixel difference map between the two corrupted variants, plus the same triplet for a blur corruption where you expect almost no difference.

**Expected outcome, stated in advance.** JPEG quantizes high-frequency DCT coefficients, so it should measurably attenuate noise corruptions (making the benchmark slightly understate pure noise severity) while having near-zero effect on blur, brightness, or contrast. The interesting deliverable is the heterogeneity across the 15 corruptions, not a single number.

**Why it is exceptional and where it is weak.** It is the cleanest possible control: one variable, perfect ground truth, and it targets a flaw the authors themselves acknowledge but never measure. The risk is that it reads as "small" if presented thinly. Defend against the "not enough effort" penalty by being thorough: all 15 corruptions, multiple severities, two or three JPEG qualities, three models, difference maps, and a short Fourier analysis of what the JPEG delta actually removes. Time budget: roughly 11 hours.
