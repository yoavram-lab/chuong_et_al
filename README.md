# chuong_et_al

This is the code for the simulation-based inference of Chuong et. al, 2023.
It is based on the code 

To run the inference and see its results:

1. install sbi using pip - 'pip install sbi'
2. generate 100k simulations using generate_presimulated_data_initial_beneficial.py:
    'python generate_presimulated_data.py -p 100000 -m WF -n reproduced -g Chuong_116_gens.txt'
3. train a neural network on the simulations you generated:
    'python infer_sbi_initial_beneficial.py -m WF -pd WF_presimulated_data_100000_reproduced.csv -pt WF_presimulated_theta_100000_reproduced.csv -g Chuong_116_gens.txt -s 42 -n reproduced'
    Once done, posterior should be at posteriors/posterior_reproduced.pkl
4. Go to Empirical Analysis.ipynb, change the posterior's path and run the notebook. 

For more information, you can contact Nadav Ben Nun (nadavbennun1@mail.tau.ac.il) or the Ram lab at Tel Aviv University.