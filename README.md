# BCI soft glove reproduction

Reproducing the EEG decoding core from a stroke rehabilitation paper out of NUS, and then pushing on it a bit further to see where it actually breaks.

**Paper:** Cheng, N., Phua, K.S., Lai, H.S., Tam, P.K., Tang, K.Y., Cheng, K.K., Yeow, R.C.H., Ang, K.K., Guan, C., Lim, J.H. (2020). *Brain-Computer Interface-Based Soft Robotic Glove Rehabilitation for Stroke.* IEEE Transactions on Biomedical Engineering, 67(12), 3339-3351. [https://doi.org/10.1109/TBME.2020.2984003](https://doi.org/10.1109/TBME.2020.2984003)

## What I understood from the paper

Eleven chronic stroke patients went through six weeks of hand rehab therapy. One group wore an EEG cap and used a motor imagery BCI to trigger a soft robotic glove, basically the glove only assisted a grasp when the decoder thought the patient was actually trying to move their hand. The other group used the same glove but without the BCI gating it, so the device just assisted on its own schedule. They compared outcomes between the two groups over the course of the trial.

The part that actually decides "is this patient trying to move their hand" is FBCSP, filter bank common spatial patterns. That's not incidental, two of the co-authors (Ang and Guan) are the people who originally built FBCSP and used it to win the BCI Competition IV back in 2008. So this paper is really asking whether hooking that decoding algorithm up to a physical glove helps stroke patients recover, not whether the algorithm itself works, since that part was already established.

Obviously I can't rebuild the clinical trial. No access to stroke patients, no access to their glove hardware, no access to the actual recorded EEG since it's private patient data. What I can do is take the actual decoding pipeline, the part of the paper that is a real, well specified algorithm, and rebuild it from scratch against a public dataset to see if it holds up.

## What I built

I implemented FBCSP from scratch: a bandpass filter bank, per band common spatial pattern filters, log variance features, and mutual information based feature selection across all the bands. No existing BCI library code, just numpy and scipy for the actual filtering and eigenvalue decomposition.

Since the patient EEG isn't public, I validated the implementation on BCI Competition IV Dataset 2a, which is the same public benchmark that FBCSP was originally developed and tested against. I used the left hand vs right hand motor imagery trials as the two class stand in for "intent to move this hand," trained a classifier per subject, and ran 10 fold cross validation across all 9 subjects.

Mean accuracy came out to about 82% (GaussianNB 81.7%, LDA 81.9%, SVM 82.1%), which lines up with what the FBCSP literature reports for this dataset. More importantly, the pattern of which subjects were easy and which were hard matched the known behavior of this exact dataset: subjects 2, 4 and 6 sit well below the rest at 51 to 68%, which is a well documented property of BCI IV-2a, not something specific to my code. That match mattered more to me than the raw accuracy number, because it's evidence the spatial filtering is picking up real sensorimotor rhythm structure and not just overfitting noise.

Full per subject numbers are in `results/benchmark.csv` and plotted in `results/accuracy_by_subject.png`.

## Where it actually gets interesting

The offline benchmark above only proves the algorithm can classify EEG that's already been cut into neat two second windows around a cue. That's not the real problem the paper had to solve. A device that assists a stroke patient's hand has to run continuously and decide, moment to moment, whether the patient is trying to move right now, including all the time they are just sitting there doing nothing.

So I built a pseudo online simulation. I trained the classifier the same way as before, then instead of feeding it pre cut trials, I slid a two second window across the raw continuous EEG at 5 Hz and had it make a decision at every step, using a confidence threshold and a small debounce rule so a single noisy window couldn't count as a trigger. For every trial I measured whether it detected the movement correctly, how long that took, and whether it falsely triggered during the two seconds of quiet rest right before the cue even appeared.

The result surprised me a bit. The plain classifier detected the correct movement 88% of the time with under half a second of latency, which sounds great. But it also falsely triggered during rest on 79% of trials. That's because a forced binary classifier has no way to output "neither," the probabilities for left hand and right hand always sum to one, so on pure rest data it just confidently picks whichever class happens to look slightly closer. The offline accuracy number never shows this because the offline evaluation never gives the model any rest data to fail on. This is a known problem in real BCI systems, usually called the idle state problem, and it only shows up once you actually simulate continuous use.

To fix it, I added a second FBCSP and LDA stage trained specifically to tell rest apart from motor imagery, using the couple of seconds right before each training cue as negative examples, and put it in front of the left/right classifier as a gate. Both stages now have to agree confidently before a trigger counts.

That gate brought the false trigger rate down from 79% to about 13%, roughly a six fold drop, and it held up consistently across all nine subjects. But it wasn't free: detection rate dropped from 88% to 68%, and latency went up from about half a second to 0.78 seconds. That's a real precision and recall tradeoff, not something I tried to hide or average away. For an assistive glove I'd actually lean toward the gated version, a false clench when the patient didn't ask for one seems worse than occasionally having to try again, but I think it's more honest to show both numbers than to just report the flattering one.

Full per subject comparison is in `results/online_simulation.csv` and plotted in `results/online_simulation_comparison.png`.
<img width="1800" height="750" alt="image" src="https://github.com/user-attachments/assets/274be1e2-a440-441c-a084-dce74644808f" />


## What this does and doesn't prove

This confirms the FBCSP decoding core is implemented correctly and behaves the way the published literature says it should, and it surfaces a real limitation (the idle state problem) that the original offline evaluation style doesn't expose. It does not validate the paper's actual clinical claims about stroke recovery outcomes, since that would need their patient data and their physical hardware, neither of which exist outside their lab.

## Repo layout

```
src/bci_glove/       core implementation: data loading, filter bank, FBCSP, cross validation, pseudo online simulation
scripts/             entry points that actually run things and save results/plots
tests/               unit tests against synthetic signals with known structure
results/             csv results and png plots from the runs described above
```

## Running it

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py           # downloads BCI IV-2a on first run, evaluates all 9 subjects offline
python scripts/plot_results.py            # writes results/accuracy_by_subject.png
python scripts/run_online_simulation.py   # baseline vs gated pseudo online simulation, all 9 subjects
python scripts/plot_online_simulation.py  # writes results/online_simulation_comparison.png
pytest tests/                             # unit tests on synthetic data
```
