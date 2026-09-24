# corroborate.rs: two runs, fingerprint by fingerprint

`xks corroborate A B` joins two `eval --rows` files on (case, question id)
and reports, over the label probabilities, the largest and mean absolute
difference, how often the argmax agrees, and each run's accuracy. `--pairs`
lists every pair.

Its use is the sites: the same subject and cases on `x86` and on `cards` or
`avx512`, where any difference is the site's arithmetic (the cards' float16
activations, a different summation order, the host's rows through another
kernel) and not the model. Read a difference against the subject's own
noise floor (polygraph): below it, the sites agree.
