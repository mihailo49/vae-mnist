# VAE na MNIST-u (Veštačka inteligencija sa primenama – domaći zadatak)

Generisanje rukom pisanih cifara pomoću Variational Autoencoder-a (PyTorch), sa cross-validacijom,
poređenjem više konfiguracija i evidencijom eksperimenata u MLflow-u.

## Struktura
```
src/data.py          učitavanje, validacija, normalizacija, podela na foldove
src/model.py         arhitekture (MLPVAE, ConvVAE)
src/train.py         gubitak, treniranje, evaluacija, vreme/resursi
src/reporting.py     agregacija rezultata
src/viz.py           slike (rekonstrukcije, uzorci, latentni prostor...)
configs/*.json       jedna konfiguracija po fajlu (hiperparametri)
notebooks/01_eda.ipynb   analiza skupa podataka
run_experiments.py   cross-validacija svih konfiguracija + MLflow
make_report.py       Excel tabele i grafikoni
final_eval.py        najbolja konfiguracija na celom skupu + test + slike
tests/test_smoke.py  brzi test celog toka
results/             CSV logovi, rezultati.xlsx, slike
```

## Instalacija
```
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Korišćenje
```
python -m pytest tests                      # brza provera da sve radi (bez interneta)
jupyter notebook notebooks/01_eda.ipynb     # analiza podataka (preuzima MNIST u data/raw)
python run_experiments.py --subset 20000    # cross-validacija svih konfiguracija (brzo)
python run_experiments.py                   # isto, na celom trening skupu
python make_report.py                       # results/rezultati.xlsx + results/figures/*.png
python final_eval.py                        # najbolja konfiguracija -> test skup + slike
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Dodatne opcije: `--configs ime1 ime2`, `--folds 3`, `--epochs 5`, `--device cpu|cuda`.

## Metodologija
- **Podaci:** MNIST (60.000 trening / 10.000 test), normalizacija na [0, 1]; izveštaj o validaciji u `results/data_report.json`.
- **Model:** VAE (encoder → μ, log σ² → reparametrizacija → decoder); gubitak = BCE rekonstrukcija + β · KL.
- **Cross-validacija:** stratifikovani k-fold na trening skupu; test skup se koristi samo u `final_eval.py`.
- **Poređenje konfiguracija:** metrika je negativni ELBO (rekonstrukcija + KL uz β = 1), jer je jedino ona uporediva između različitih β.
- **Reproduktivnost:** fiksirani seed-ovi (`seed + redni broj folda`), verzije biblioteka u `results/environment.json` i u MLflow tagovima.
- **Evidencija:** hiperparametri, gubitak po epohi, metrike po foldu, vreme treniranja/inferencije, veličina modela i RAM u MLflow-u i u `results/*.csv`.

## Konfiguracije
| Ime | Model | Latent | β | LR |
|---|---|---|---|---|
| mlp_z16_base | MLP | 16 | 1 | 1e-3 |
| mlp_z2 | MLP | 2 | 1 | 1e-3 |
| mlp_z32 | MLP | 32 | 1 | 1e-3 |
| mlp_z16_beta4 | MLP | 16 | 4 | 1e-3 |
| mlp_z16_lr3e-4 | MLP | 16 | 1 | 3e-4 |
| conv_z16 | Conv | 16 | 1 | 1e-3 |

## Rezultati
Projekat je u izradi. Rezultati eksperimenata (tabele, grafikoni i zaključak) biće dodati nakon završetka treniranja.
