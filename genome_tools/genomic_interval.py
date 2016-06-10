# Copyright 2015 Jeff Vierstra

import numpy as np

class genomic_interval(object):

	def __init__(self, chrom, start, end, name = '.', score = None, strand = None):
		self.chrom = str(chrom)
		self.start = int(start)
		self.end = int(end)
		self.name = str(name)
		self.score = score
		self.strand = strand
		
	def __len__(self):
		return self.end - self.start
	
	def __str__(self):
		return '\t'.join( [ str(x) for x in [self.chrom, self.start, self.end] ] )

	def widen(self, w):
		return genomic_interval(self.chrom, self.start - w, self.end + w, self.name, self.strand)

class genomic_interval_set(object):

	def __init__(self, iterator = []):
		self.intervals = []
		for interval in iterator:
			self.intervals.append(interval)
	
	def __len__(self):
		return len(self.intervals)

	def __iter__(self):
		for x in self.intervals:
			yield x

	def __getitem__(self, i):
		return self.intervals[i]

	def __add__(self, other):
		if type(other) == genomic_interval_set:
			self.intervals.extend(other.intervals)
		else:
			self.intervals.append(other)

def bin_intervals(intervals, bins):
	"""Bin genomic intervals by score thresholds

	Args:
		intervals (genomic_interval_set): a set of genomic intervals
						that contain a score value
		bins (array): a list of values comprising desired bins
						(must by monotonically increasing or decreasing)
	Returns:
		(list): a list of indices for each input interval to a corresponding bin
	"""
	scores = np.zeros(len(intervals))
	for i in range(len(intervals)):
		scores[i] = intervals[i].score
	return np.digitize(scores, thresh)