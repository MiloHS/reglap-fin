# reglap-fin
This repository contains code for learning financial graphs based on asset correlations over time while incorporating of regressors.

## 🚀 Features

- Builds correlation based dissimilarity matrices from a data array
- Learns financial graphs from log-returns and dissimilarity matrices through smoothness and a k-lag causal graph process.
- Incorporates regressors in signal estimation and learns their weights over time.

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/MiloHS/reglap-fin.git

# Navigate into the project directory
cd MiloHS/reglap-fin

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.tx
