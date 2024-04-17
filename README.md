# chuong_et_al

This is the code for the simulation-based inference of the paper:
> Julie N. Chuong, Nadav Ben Nun, Ina Suresh, Julia Matthews, Titir De, Grace Avecilla, Farah Abdul-Rahman, Nathan Brandt, Yoav Ram, David Gresham (2024) _DNA replication errors are a major source of adaptive gene amplification._

It shares some components with the code of [Avecilla et al. (2022)](https://doi.org/10.1371/journal.pbio.3001633) ([graceave/cnv_sims_inference](https://github.com/graceave/cnv_sims_inference)), mostly regarding network training.

To run the inference and see its results:

1. Install `sbi` library: `pip install sbi`
2. Create a folder named `presimulated_data`, and generate 100k simulations:
    `python generate_presimulated_data.py -p 100000 -m WF -n reproduced -g Chuong_116_gens.txt`
3. Train a neural density estimator on the simulations:
    `python infer_sbi_initial_beneficial.py -m WF -pd WF_presimulated_data_100000_reproduced.csv -pt WF_presimulated_theta_100000_reproduced.csv -g Chuong_116shares_gens.txt -s 42 -n reproduced`
    Once done, posterior should be at `posteriors/posterior_reproduced.pkl`
4. Go to `Empirical Analysis.ipynb`, change the posterior path, and run the notebook.

To run the collective posterior inference, run the `Overall Posterior.ipynb` notebook.
- The normalizing constants are saved in `posteriors/log_Cs`, but can be re-generated using `OverallPosterior.get_log_C` method.
- Overall posterior MAPs and samples are in the `maps` folder, but can be re-generated using `get_map` and `sample` methods.

To review your trained network and simulations, you can go to `Simulation Analysis.ipynb`.
These simulations were generated from a narrow parameter range (details in the notebook).
If you wish to validate your trained network on a different simulation set, you can simply generate other simulations by following step 2 above, and change the relevant cells in the notebook.

For more information, you can contact [Nadav Ben Nun](mailto:nadavbennun1@mail.tau.ac.il) or the [Ram lab](https://www.yoavram.com) at Tel Aviv University.

## License 

Source code: MIT License.
Other content: Creative Commons Attribution 4.0 International License.
