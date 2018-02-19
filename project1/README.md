# Project 1

### CS 7642 Spring 2018

### Dan Frakes (dfrakes3)

To run Project 1 code, clone the repository and run `python main.py` from the `project1` subdirectory.

By default, this script will run the code necessary to plot and display replicas of Figures 3, 4, and 5 from Sutton's paper.

Comment out the code between docstrings in `main.py` in order to omit that figure.  Please note that Figure 5 relies on the results from Figure 4 computation (though there are "cached" best alpha values, it is best to re-run this code in case there are any tweaks to the alpha values being tested).

Additionally, hyperparameters can be adjusted in `settings/__init__.py`.  Variable names that are capitalized are global and should be self-explanatory in the context of this assignment.
