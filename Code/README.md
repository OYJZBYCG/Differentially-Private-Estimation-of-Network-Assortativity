# Paper Experiments

Run the privacy protocols on the input graphs to estimate assortativity and compute mean squared error (MSE) and mean absolute relative error (MARE). Use the results from these runs to generate the paper figures. All three methods use the final estimator in assortativity.py. The code generates the Barabasi-Albert (BA) graphs, and the two datasets.csv manifests specify the inputs. The original edge lists for the real graphs are in data/.

| Experiment | Python module | Paper figures |
|---|---|---|
| Privacy budget | privacy_budget.run | Figs. 2, 5, and 6 |
| BA graph size | graph_scale.run | Fig. 3 |
| Graph sparsity | graph_sparsity.run_ba and graph_sparsity.run_real | Fig. 4 |

## Run the experiments

Select a working local Python interpreter in PyCharm and open a PowerShell terminal in code/. The commands write results to paper_runs/ alongside code/ and run 100 trials per graph and parameter setting. To resume a run from its checkpoints, use the same settings and output directory. The largest BA graph has about five billion candidate vertex pairs.

~~~powershell
python -m pip install -r .\requirements.txt

python -m privacy_budget.run --suite ablation --manifest .\datasets.csv --output ..\paper_runs\correction
python -m privacy_budget.run --suite allocation --manifest .\datasets.csv --output ..\paper_runs\allocation

python -m graph_scale.run --output ..\paper_runs\scale
python -m graph_sparsity.run_ba --output ..\paper_runs\ba_sparsity

python -m graph_sparsity.run_real --keep "0.2,0.4,0.6,0.8,1" --workers 4 --output ..\paper_runs\real_sparsity
python -m graph_sparsity.run_real --keep "0.05,0.1" --workers 4 --output ..\paper_runs\real_low_retention
~~~

## Generate the figures

~~~powershell
python -m privacy_budget.plot --correction ..\paper_runs\correction\summary.csv --allocation ..\paper_runs\allocation\summary.csv --output ..\paper_runs\figures
python -m graph_scale.plot --trials ..\paper_runs\scale --output ..\paper_runs\figures
python -m graph_sparsity.plot --ba-trials ..\paper_runs\ba_sparsity --real-trials ..\paper_runs\real_sparsity ..\paper_runs\real_low_retention --output ..\paper_runs\figures
~~~

The graph size and sparsity plotting modules read the saved estimates and average MSE and MARE across graph instances with equal weights. Use --save-data to export the plotting table to a CSV file. Add --data-only to export the table without generating the figure.
