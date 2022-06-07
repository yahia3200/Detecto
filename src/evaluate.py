'''
               Cairo University 
            Faculty of Engineering

    Pattern Recognition and Neural Networks 
            Spring 2022

    --------------------------------------------------------
      Handwriting Based Gender Classification
    -------------------------------------------------------

    Evaluation Script
'''

import numpy as np 

'''--Read submission.readme before you run or update this script--'''

OUTPUT_DIRECTORY = "./out/" ## * Appended Path of outputs

## * Classification Evaluation:
truth = None
with open('./ground_truth.txt', 'rb') as gt_file:
    truth = [ int(line)  for line in gt_file.readlines()]
    truth = np.array(truth)

hypothesis = None
with open(OUTPUT_DIRECTORY + "results.txt", 'rb') as hypo_file:
    hypothesis = [ int(line)  for line in hypo_file.readlines()]
    hypothesis = np.array(hypothesis)

## ! Account for length mismatch:
if len(truth) != len(hypothesis):
    truncation_len = min(len(truth), len(hypothesis))
    hypothesis = hypothesis[:truncation_len]
    truth = truth[:truncation_len]

ACCUARICY = np.sum(truth == hypothesis) / len(truth)    

## * Time Evaluation:
times = None
with open(OUTPUT_DIRECTORY + "times.txt", 'rb') as times_file:

    times = [float(line) for line in times_file.readlines()]
    times = np.array(times)

TIME_MEAN = round(np.mean(times), 3)


## * Report:
print(ACCUARICY*100, TIME_MEAN)
