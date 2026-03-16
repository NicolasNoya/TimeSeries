# Time Series Classification with Deep Learning

This project explores various approaches for Time Series Classification, focusing on the **LSST (Large Synoptic Survey Telescope)** dataset from the UEA benchmark. It ranges from basic exploratory data analysis and classical machine learning baselines to modern, transformer-based Time Series Foundation Models.

## Dataset

The **LSST dataset** consists of multispectral time-series data featuring simulated light curves of transient astronomical events across 6 different passbands ($u, g, r, i, z, y$). The task is to accurately classify these objects into their respective astronomical classes based on their temporal signal. 

## Project Structure

The project is organized into modular Jupyter Notebooks, each focusing on a distinct modeling approach:

### 1. Data Exploration & Baselines
* **`eda.ipynb`**: Exploratory Data Analysis. Analyzes the LSST dataset distributions, visualizes light curves, and explores class imbalances.
* **`baseline.ipynb`**: Simple Machine Learning baselines classifying light curves using simple flat arrays and basic statistical summaries (mean, std, min, max, CoV).
* **`handcrafted_features.ipynb`**: Advanced feature extraction leveraging domain-specific libraries (e.g., `light-curve`) to capture complex temporal phenomena before passing them to classical classifiers.

### 2. Time Series Foundation Models
These notebooks evaluate state-of-the-art pre-trained time-series models and representations via Linear Probing:
* **`chronos.ipynb`**: Evaluates the [Chronos](https://github.com/amazon-science/chronos-forecasting) family of models (e.g., `autogluon/chronos-2-small`) by extracting embeddings and probing performance.
* **`moment.ipynb`**: Explores the [MOMENT](https://moment.cs.cmu.edu/) foundation model (`AutonLab/MOMENT-1-large`), using its learned representations for the sequence classification task. 
* **`mantis.ipynb`**: Evaluates **Mantis** (specifically `Mantis8M`), a lightweight representation model targeted for Time Series Classification. 

## Setup and Requirements
To run these notebooks, ensure you have standard data science libraries installed, along with `tslearn` for fetching the UEA datasets. 
Additionally, for foundation models, you may need specifically:
* `torch` and `transformers` 
* `amazon-chronos` / `autogluon`
* `momentfm` 
* `mantis`
* `light-curve` (for domain specific handcrafted features)
