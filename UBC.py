import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import math

df = pd.read_csv('Ads_CTR_Optimisation.csv')

print(df.shape)

print(df.describe())

N = 1000
d = 10

ads_selected = []*d