import numpy as np

'''
#######################################################################################################################
#######################################################################################################################
The categorical.py module computes categorical verification from the two inputs (mostly observed and model data).
There are eight types of the categorical verification that can be used:

- prop_cor (Propotion Correct)
- FA_Ratio (False Alarm Ratio)
- UER (Undected Error Rate)
- HR (Hit Rate)
- FA_Rate (False Alarm Rate)
- BS (Bias Score)
- TS (Threat Score)
- ETS (Equitable Threat Score)
#######################################################################################################################
#######################################################################################################################

'''

class Categorical:

	''' computes the binary event depending on the threshold input. Then, calculate the hits, misses, false alarms and
		correct rejections. Finally, call category_type and get the categorical verification. 
	'''
	def categorical(self, obs, model, threshold, cat_type):
		dich_model = (model >= threshold)
		dich_obs = (obs >= threshold)

		hits = np.sum(np.bitwise_and(dich_model == True, dich_obs == True, dtype=int))
		misses = np.sum(np.bitwise_and(dich_model == False, dich_obs == True, dtype=int))
		false_alarms = np.sum(np.bitwise_and(dich_model == True, dich_obs == False, dtype=int))
		correct_rej = np.sum(np.bitwise_and(dich_model == False, dich_obs == False, dtype=int))

		if len(np.unique(dich_model)) == 1:
			if np.unique(dich_model[0])[0] == False or np.unique(dich_obs[0])[0] == False:
				return np.nan
			else:
				return self.category_type(hits, misses, false_alarms, correct_rej, cat_type)
		else:		
			return self.category_type(hits, misses, false_alarms, correct_rej, cat_type)

	# return the calculated categorical type 
	def category_type(self, hits, misses, false_alarms, correct_rej, category_type):
		if category_type == 'prop_cor':
			return (false_alarms)/(hits + false_alarms.astype(float))

		elif category_type == 'FA_Ratio':
			return (false_alarms)/(hits + false_alarms.astype(float))

		elif category_type == 'UER':
			return (misses)/(hits + misses.astype(float))

		elif category_type == 'HR':
			return (hits)/(hits + misses.astype(float))

		elif category_type == 'FA_Rate':
			return (false_alarms)/(false_alarms + correct_rej.astype(float))

		elif category_type == 'BS':
			return (hits + false_alarms)/(hits + misses.astype(float))

		elif category_type == 'TS':
			return (hits)/(hits + false_alarms + misses.astype(float))

		elif category_type == 'ETS':
			dich_total = hits + misses + false_alarms + correct_rej
			CRF = (hits + misses)/dich_total.astype(float)
			Sf = CRF*(hits + false_alarms.astype(float))
			ETS = (hits - Sf)/(hits + false_alarms + misses - Sf.astype(float))

			return ETS 

	# get the name of the categorical verification method from the input category_type
	def category_title(self, category_type):
		if category_type == 'prop_cor':
			return 'Proportion Correct'

		elif category_type == 'FA_Ratio':
			return 'False alarm Ratio'

		elif category_type == 'UER':
			return 'Undetected Error Rate'

		elif category_type == 'HR':
			return 'Hit Rate'

		elif category_type == 'FA_Rate':
			return 'False Alarm Rate'

		elif category_type == 'BS':
			return 'Bias Score'

		elif category_type == 'TS':
			return 'Threat Score'

		elif category_type == 'ETS':
			return 'Equitable Threat Score' 

	# get the maximum value of the certain category verification method
	def category_max(self, category_type):
		if category_type in ['prop_cor','FA_Ratio','UER','HR','FA_Rate','TS','ETS']:
			return 1.2

		elif category_type == 'BS':
			return 2.2

	# get the minimum value of the certain category verification method
	def category_min(self, category_type):
		if category_type in ['prop_cor','FA_Ratio','UER','HR','FA_Rate','TS']:
			return 0

		elif category_type == 'BS':
			return 0

		elif category_type == 'ETS':
			return -1/3