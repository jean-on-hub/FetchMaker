# Import libraries
import numpy as np
import pandas as pd
import codecademylib3
from scipy.stats import binom_test
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency
# Import data
dogs = pd.read_csv('dog_data.csv')

# Subset to just whippets, terriers, and pitbulls
dogs_wtp = dogs[dogs.breed.isin(['whippet', 'terrier', 'pitbull'])]

# Subset to just poodles and shihtzus
dogs_ps = dogs[dogs.breed.isin(['poodle', 'shihtzu'])]

# inspect data
# print(dogs.head())
# using binomial test to  know if whippets are significantly more or less likely than other dogs to be a rescue.
whippet_rescue = dogs["is_rescue"][dogs.breed == "whippet"]
num_whippet_rescues = np.sum(whippet_rescue == 1)
print(num_whippet_rescues)
num_whippets = len(dogs[dogs.breed == "whippet"])
print(num_whippets)
pvalue = binom_test(6, 100,p =0.08)
print(pvalue)
# with this pvalue we can say  8% of whippets are rescues

# using anova to check if  there is a significant difference in the average weights of these three dog breeds
wt_whippets = dogs.weight[dogs.breed == "whippet"]
wt_terriers = dogs.weight[dogs.breed == "terrier"]
wt_pitbulls = dogs.weight[dogs.breed == "pitbull"]
# print(wt_pitbulls)
ans1 ,pvalue =f_oneway(wt_whippets,wt_terriers,wt_pitbulls)
print(pvalue)
# from this pvalue at least one pair of dog breeds have significantly different average weights.
# using tukeys to determine the reltionships

tukey_results = pairwise_tukeyhsd(endog = dogs_wtp.weight,groups = dogs_wtp.breed)
print(tukey_results)
# using chi2 to determine if There is an association between breed (poodle vs. shihtzu) and color
Xtab = pd.crosstab(dogs_ps.breed,dogs_ps.color)
chi2, pval, dof, exp = chi2_contingency(Xtab)

print(pval)

