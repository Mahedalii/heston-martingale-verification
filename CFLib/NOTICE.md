# NOTICE — CFLib

`CFLib` is the computational finance teaching library of **Prof. Pietro Rossi**
(Department of Statistical Sciences "Paolo Fortunati", University of Bologna),
distributed to students of the Computational Finance course.

It is **not** the work of this repository's author.

It is included here, unmodified, solely so that `heston_martingales.py` runs without
requiring a separate download. Only the eight modules actually imported by this project
are included; the remainder of the library has been omitted.

| Module | Role in this project |
|---|---|
| `config.py` | command-line parameter parsing |
| `Heston.py` | Heston model object |
| `FT_opt.py` | SINC Fourier option pricing (`ft_opt`) |
| `cir_obj.py` | CIR variance process object |
| `cir_evol.py` | quadratic-exponential CIR simulation (`QT_cir_evol`) |
| `heston_evol.py` | exact conditional spot evolution (`mc_heston`) |
| `stats.py` | mean and standard deviation helper |
| `IO_utils.py` | array display helper (imported by `cir_obj`) |

The SINC pricing algorithm implemented in `FT_opt.py` is published in:

> F. Baschetti, G. Bormetti, S. Romagnoli and P. Rossi.
> *The SINC way: a fast and accurate approach to Fourier pricing.*
> Quantitative Finance, 22(3): 427–446, 2022. DOI: 10.1080/14697688.2021.1965192

No licence is attached to the original library. It is reproduced here in good faith for
reproducibility, with attribution. **If Prof. Rossi would prefer it not be redistributed,
please contact me and I will remove it immediately**, replacing it with setup instructions
pointing to the course materials.