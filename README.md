# Resistance Exercises Classification #
Classification of twelve different resistance exercises using cosine similarity and support vector machine:

0. Bench press
1. Shoulder press
2. Bicep curl
3. Upright rows
4. Lateral raises
5. Overhead triceps extensions
6. Kneeling triceps kickbacks
7. Standing bent-over rows
8. Kneeling bent-over rows
9. Squats
10. Forward walking lunges
11. Calf raises

## Language ##
[Python 2.7.x](https://www.python.org/)

## Packages ##
1. [Numpy](http://www.numpy.org/)
2. [Scipy](https://www.scipy.org/)
3. [Matplotlib](http://matplotlib.org/)
4. [Sklearn](http://scikit-learn.org/stable/)

## Data Sample ##
0,-0.886,-0.487,0.117,0

1st column: time

2nd column: X axis data

3rd column: Y axis data

4th column: Z axis data

5th column: label (0-11)

## How to Run ##
For cosine similarity:

  *python cosineSim_rc.py training_set.csv /testing_set_folder/*

For support vector machine:

  *python svm_rc.py /folder_containing_all_datasets/*

## Contact ##
junguo AT boisestate.edu

## References ##
1. https://github.com/demotu/BMC

## License ##
GPL v3 license
