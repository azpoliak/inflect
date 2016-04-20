#!usr/bin/env python
from operator import itemgetter

# David Russell / Adam Poliak 
# Modification of NLP Viterbi assignment
# to do inflection morphology prediction with
# part of speech tagging

import argparse
import pdb
import numpy as np
from itertools import izip
import os
import sys
import codecs
import string


#set of possible tags for each word
tag_dict = {}
#number of times word w gets tagged with tag t
count_wt = {}
#bigram models of tags
count_tt = {}
#bigram models of word
count_w = {}
# number of times each tag appears
count_tag = {}
# sum(count_tag.values())
total_tag_count = 0.0
# best pprobability path from the begining to state i with tag t
mu = {}
# backpointer of tag and i to previous tag
backpointers = {}
#set of all training tokens
train_words_set = set()
#count of all words in the vocab
total_vocab_size = 0.0
# size of training data
total_train_size = 0.0
#singletons
sing_tt = {}
sing_wt = {}
# alpha dictionary
alphas = {}
# beta dictionary
betas = {}

def compute_ptt(prev_tag, tag, total_tag_count, ec, lmbd):
	if ec:
		if prev_tag not in sing_tt:
			sing_tt[prev_tag] = 0
		lmbd = sing_tt[prev_tag] + 1
		if (prev_tag, tag) not in count_tt:
			count_tt[(prev_tag, tag)] = 0
		numerator = count_tt[(prev_tag, tag)] + (lmbd * (count_tag[tag] / float(total_train_size)))
		denom = count_tag[prev_tag] + lmbd
		return np.log(numerator / float(denom))

	lmbdV = lmbd*len(count_tag)
	pt = (count_tag[tag] + lmbd)/float(total_tag_count+lmbdV)
	if (prev_tag, tag) not in count_tt:
		count_tt[(prev_tag, tag)] = 0
	toReturn = (count_tt[(prev_tag, tag)] + lmbdV*pt) / float(count_tag[prev_tag]+lmbdV)
	if toReturn <= 0.0:
		toReturn = (count_tag[tag]/float(total_train_size))
	return np.log(toReturn)

def compute_ptw(word, tag, total_vocab_size, total_word_count, train_vocab_size, ec, lmbd):
	if ec:
		if tag not in sing_wt:
			sing_wt[tag] = 0
		lmbd = sing_wt[tag] + 1
		if (word, tag) not in count_wt:
			count_wt[(word,tag)] = 0
		numerator = count_wt[(word, tag)] + (lmbd * (count_w[word] + 1) / (total_train_size + train_vocab_size))
		denom = count_tag[tag] + lmbd
		return np.log(numerator / float(denom))

	lmbdV = lmbd*total_vocab_size
	pw = (count_w[word] + lmbd)/float(total_word_count + lmbdV)
	toReturn = (count_wt[(word, tag)] + lmbdV*pw) / float(count_tag[tag] + lmbdV)
	if toReturn <= 0.0:
		toReturn = (count_w[word]/float(total_word_count))
	return np.log(toReturn)

def get_mu(tag, i):
	if (tag, i) not in mu:
		return -1 * np.inf
	return mu[(tag, i)]

def get_alpha(tag, i):
	if (tag, i) not in alphas:
		return -1 * np.inf
	return alphas[(tag, i)]

def get_beta(tag, i):
	if (tag, i) not in betas:
		return -1 * np.inf
	return betas[(tag, i)]

def log_add(x, y):
	# if x is -np.inf or y is -np.inf:
	# 	return -np.inf
	if y <= x:
		return x + np.log(1 + np.exp(y - x))
	expr = y + np.log(1 + np.exp(x-y))
	return expr

def sorted_dict(d, exp):
	if exp is True:
		blah = sorted([(x[0],np.exp(x[1])) for x in d.iteritems()], key=itemgetter(1))
	else:
		blah = sorted([(x[0],(x[1])) for x in d.iteritems()], key=itemgetter(1))
	return blah


def utf8read(file): return codecs.open(file, 'r', 'utf-8')

def train(t, l, w, pos_file):
	'''
	Read the train data and store the counts in global tables
	'''
	def combine(a, b): return '%s.%s' % (a, b)
	# Build the LEMMAS hash, a two-level dictionary mapping lemmas to inflections to counts
	prev_word = '###'
	prev_tag = '###'
	curr_sentence_count = 1
	number_of_sentences = len(utf8read(combine(t,w)).readlines())
	for words, lemmas, posses in izip(utf8read(combine(t, w)), utf8read(combine(t, l)), utf8read(combine(t, pos_file))):
		delim = unicode(' ### ', 'utf-8')
		words = delim + words
		lemmas = delim + lemmas
		posses = delim + posses
		if curr_sentence_count == number_of_sentences:
			posses = string.replace(posses, '\n', delim + '\n')
			lemmas = string.replace(lemmas, '\n', delim +'\n')
			words = string.replace(words, '\n', delim + '\n')
		#print words
		curr_sentence_count += 1
		for word1, lemma, pos in izip(words.rstrip().lower().split(), lemmas.rstrip().lower().split(),  posses.rstrip().split()):
                        tag = word1
			word = pos + '&!&' + lemma
			global total_train_size
			total_train_size += 1
			if word not in tag_dict:
				tag_dict[word] = set()

			tag_dict[word].add(tag)
			if (word, tag) not in count_wt:
				count_wt[(word, tag)] = 0
			count_wt[(word, tag)] += 1
			if word not in count_w:
				count_w[word] = 0
			count_w[word] += 1


			
			if count_wt[(word, tag)] == 1:
				if tag not in sing_wt:
					sing_wt[tag] = 0
				sing_wt[tag] += 1
			elif count_wt[(word, tag)] == 2:
				sing_wt[tag] -= 1
			
			if (prev_tag, tag) not in count_tt:
				count_tt[(prev_tag, tag)] = 0
			count_tt[(prev_tag, tag)] += 1
			
			if count_tt[(prev_tag, tag)] == 1:
				if prev_tag not in sing_tt:
					sing_tt[prev_tag] = 0
				sing_tt[prev_tag] += 1
			elif count_tt[(prev_tag, tag)] == 2:
				sing_tt[prev_tag] -= 1
			
			if tag not in count_tag:
				count_tag[tag] = 0
			count_tag[tag] += 1

			prev_word = word
			prev_tag = tag
                        
			if lemma not in tag_dict:
				tag_dict[lemma] = set()
			tag_dict[lemma] = tag_dict[lemma].union(tag_dict[word])
			
	count_tag['###'] -= 1
	count_wt[('###&!&###', '###')] -= 1
	count_w['###&!&###'] -= 1
	#import pdb
	#pdb.set_trace()


def viterbi(words, ec, lmbd):
	words.append(unicode('###&!&###', 'utf-8'))
	total_vocab_size = len(train_words_set.union(set(words)))
	total_tag_count = sum(count_tag.values())
	total_word_count = sum(count_w.values())
	train_vocab_size = len(train_words_set)
	n = len(words) - 1
	mu[('###', 0)] = np.log(1.0)
	numOOV = len([x for x in words if x not in train_words_set])
	for i in range(1, len(words)):
		if words[i] not in tag_dict:
			if words[i].split('&!&')[1] in tag_dict:
				#sys.stderr.write("\n poop1 \n" + words[i] + " "  +  str(len( tag_dict[words[i].split('&!&')[1]])) + '\n'  )
				tag_dict[words[i]] = tag_dict[words[i].split('&!&')[1]]
				count_w[words[i]] = numOOV
			else:
				#sys.stderr.write('\n' + words[i] + '\n')
				tag_dict[words[i]] = set()
				tag_dict[words[i]].add(words[i].split('&!&')[1])
				#sys.stderr.write(' '.join(tag_dict[words[i]]))
				#sys.stderr.write('\n poop2 \n \n') 
				count_tag[words[i].split('&!&')[1]] = 1	
				count_w[words[i]] = numOOV
				'''
				tag_dict[words[i]] = set(count_tag.keys())
				#tag_dict[words[i]].remove('###')
				count_w[words[i]] = numOOV
				'''
				#sys.stderr.write("\n REALLY poop \n" + words[i] + str(len(count_tag.keys())) + '\n')
		for tag in tag_dict[words[i]]: #latex symbol
			if (words[i], tag) not in count_wt:
				count_wt[(words[i], tag)] = 0
			for prev_tag in tag_dict[words[i-1]]:
				if (prev_tag, tag) not in count_tt:
					count_tt[(prev_tag, tag)] = 0
				# Using natural log.
				l_ptt = compute_ptt(prev_tag, tag, total_tag_count, ec, lmbd)
				l_ptw = compute_ptw(words[i], tag, total_vocab_size, total_word_count, train_vocab_size, ec, lmbd)
				l_p = l_ptt + l_ptw
				u = get_mu(prev_tag, i-1) + l_p
				if u > get_mu(tag, i):
					mu[(tag, i)] = u
					backpointers[(tag, i)] = prev_tag

	curr_tag = unicode('###', 'utf-8')
	guessed_tags = [curr_tag]
	for i in range(len(words)-1, 0, -1):
		curr_tag = backpointers[(curr_tag, i)]
		guessed_tags.append(curr_tag)

	guessed_tags.reverse()
	return guessed_tags

def forward_backward(words, ec, lmbd):
	numOOV = len([x for x in words if x not in train_words_set])
	delim = unicode(' ### ', 'utf-8')
	words.append(unicode('###&!&###', 'utf-8'))
	total_vocab_size = len(train_words_set.union(set(words)))
	total_tag_count = sum(count_tag.values())
	total_word_count = sum(count_w.values())
	train_vocab_size = len(train_words_set)
	alphas[(delim, 0)] = np.log(1.0)
	for i in range(1, len(words)):
		if words[i] not in tag_dict:
                        if words[i].split('&!&')[1] in tag_dict:
                                #sys.stderr.write("\n poop1 \n" + words[i] + " "  +  str(len( tag_dict[words[i].split('&!&')[1]])) + '\n'  )
                                tag_dict[words[i]] = tag_dict[words[i].split('&!&')[1]]
                                count_w[words[i]] = numOOV
                        else:
                                #sys.stderr.write('\n' + words[i] + '\n')
                                tag_dict[words[i]] = set()
                                tag_dict[words[i]].add(words[i].split('&!&')[1])
                                #sys.stderr.write(' '.join(tag_dict[words[i]]))
                                #sys.stderr.write('\n poop2 \n \n')
                                count_tag[words[i].split('&!&')[1]] = 1
                                count_w[words[i]] = numOOV
                                '''
                                tag_dict[words[i]] = set(count_tag.keys())
                                #tag_dict[words[i]].remove('###')
                                count_w[words[i]] = numOOV
                                '''
                                #sys.stderr.write("\n REALLY poop \n" + words[i] + str(len(count_tag.keys())) + '\n')

		for tag in tag_dict[words[i]]:
			for prev_tag in tag_dict[words[i-1]]:
				l_ptt = compute_ptt(prev_tag, tag, total_tag_count, ec, lmbd)
				l_ptw = compute_ptw(words[i], tag, total_vocab_size, total_word_count, train_vocab_size, ec, lmbd)
				l_p = l_ptt + l_ptw
				alphas[(tag, i)] = log_add(get_alpha(tag, i), get_alpha(prev_tag, i-1) + l_p)
	

	# In the pseduo-code, the previous loop ended at n and here we set S to alphas[(###, n)]
	S = get_alpha(delim, len(words)-1)
	alphaBeta = {}
	betas[(delim, len(words)-1)] = np.log(1.0)
	for i in range(len(words)-1, 0, -1):
		for tag in tag_dict[words[i]]:
			alphaBeta[(tag, i)] = get_alpha(tag, i) + get_beta(tag, i) - S
			for prev_tag in tag_dict[words[i-1]]:
				l_ptt = compute_ptt(prev_tag, tag, total_tag_count, ec, lmbd)
				l_ptw = compute_ptw(words[i], tag, total_vocab_size, total_word_count, train_vocab_size, ec, lmbd)
				lp = l_ptt + l_ptw
				#print get_beta(prev_tag, i-1), lp + get_beta(tag, i)
				betas[(prev_tag, i-1)] = log_add(get_beta(prev_tag, i-1), lp + get_beta(tag, i))
				alphaBeta[(prev_tag, tag, i)] = get_alpha(prev_tag, i-1) + lp + get_beta(tag, i) - S

	curr_tag = delim
	guessed_tags = [curr_tag]
	for i in range(1, len(words)):
		best_tag_lp = -1*np.inf
		best_tag = ''
		for tag in tag_dict[words[i]]:
			if alphaBeta[(tag, i)] > best_tag_lp:
				best_tag_lp = alphaBeta[(tag, i)]
				best_tag = tag
		guessed_tags.append(best_tag)
	return guessed_tags


def main():
	PARSER = argparse.ArgumentParser(description="Inflect a lemmatized corpus")
	PARSER.add_argument("-ec", type=bool, default=False, help="Extra Credit Problem 4 tag")
	PARSER.add_argument("-lmbd", type=float, default=0.0, help="Lambda smoothing parameter")
	PARSER.add_argument("-t", type=str, default="data/train", help="training data prefix")
	PARSER.add_argument("-l", type=str, default="lemma", help="lemma file suffix")
	PARSER.add_argument("-w", type=str, default="form", help="word file suffix")
	PARSER.add_argument("-pos", type=str, default="tag", help="tag file suffix for train")
	#PARSER.add_argument("-test_pos", type=str, default="data/combotest.tag", help="tag file suffix for test")

	args = PARSER.parse_args()

	'''
    (a) Read the train data and store the counts in global tables. (Your functions for
	computing probabilities on demand, such as ptw, should access these tables. In
	problem 3, you will modify those functions to do smoothing.)
	'''

	train(args.t, args.l, args.w, args.pos)

	# creates a set of token in training data
	train_words_set = set(tag_dict.keys())

	'''
	(b) Read the test data ~w into memory.
	'''
	sys.stderr.write("READING TEST DATA \n")
	test_words = []
        test_poses = utf8read("data/train.tag")
        #input = sys.stdin.read()
        #number_of_sentences = len(input)
        curr_sentence_count = 1
	for line in sys.stdin: #input:
		sys.stderr.write("input sentence number: " + str(curr_sentence_count) + '\n')
                poses = '### ' + test_poses.readline()
                line = '### ' + line
		curr_sentence_count += 1
		for lemma, pos in izip(line.rstrip().lower().split(), poses.rstrip().split()):	
                	test_words.append(pos.decode('utf-8') + "&!&" + lemma.decode('utf-8'))
	'''
	(c) Follow the Viterbi algorithm pseudocode in Figure 2 to find the tag sequence ~t that
	maximizes p(~t, ~w).
	'''
	sys.stderr.write("DONE READING TEST DATA \n")

	global total_vocab_size
	total_vocab_size = len(train_words_set.union(set(test_words)))
	guessed_tags = viterbi(test_words, args.ec, args.lmbd)[1:]
	#guessed_tags = forward_backward(test_words, args.ec, args.lmbd)

	print guessed_tags

	sys.stderr.write("DONE VITERBI \n")

	curr_sent = []
	delim = unicode(' ### ', 'utf-8')
	for inflection in guessed_tags:
		if inflection == '###':
			#Matt's code to print
			print ' '.join(curr_sent)
			curr_sent = []
		else:
			curr_sent.append(inflection.encode('utf-8'))
		
	sys.stderr.write("NOW SCORE \n")

        '''
	pdb.set_trace()


	fb_guessed_tags = forward_backward(test_words, args.ec, args.lmbd)

	(d) Compute and print the accuracy and perplexity of the tagging. (You can compute
	the accuracy at the same time as you extract the tag sequence while following
	backpointers.)
	
	#Compare with excel sheet
	from operator import itemgetter
	blah = sorted([(x[0],(x[1])) for x in mu.iteritems()], key=itemgetter(1))
	for b in blah:
		print b
	
#	test_words_set = set(test_words)

	novel_words = set(test_words).difference(train_words_set)
	known_words = set(test_words).intersection(train_words_set)

	correct_count = 0
	fb_correct_count = 0
	novel_correct_count = 0
	known_correct_count = 0
	known_words_count = 0
	novel_words_count = 0
	fb_novel_correct_count = 0
	fb_known_correct_count = 0


	for i in range(len(test_tags)):
		if guessed_tags[i] == '###':
			continue
		if fb_guessed_tags[i] == test_tags[i]:
			fb_correct_count += 1
			if test_words[i] not in known_words:
				fb_novel_correct_count += 1
			else:
				fb_known_correct_count += 1
		if guessed_tags[i] == test_tags[i]:
			correct_count += 1
			if test_words[i] not in known_words:
				novel_correct_count += 1
			else:
				known_correct_count += 1
		if test_words[i] in novel_words:
			novel_words_count += 1
		else:
			known_words_count += 1		
	correct_accuracy = correct_count / float(len(test_tags) - test_tags.count('###'))
	fb_correct_accuracy = fb_correct_count / float(len(test_tags) - test_tags.count('###'))
	if len(novel_words) == 0:
		novel_accuracy = 0.0
		fb_novel_accuracy = 0.0
	else:
		novel_accuracy = novel_correct_count / float(novel_words_count)
		fb_novel_accuracy = fb_novel_correct_count / float(novel_words_count)
	if len(known_words) == 0:
		known_accuracy = 0.0
		fb_known_accuracy = 0.0
	else:
		known_accuracy = known_correct_count / float(known_words_count) 
		fb_known_accuracy = fb_known_correct_count / float(known_words_count)

	print ("Tagging accuracy (Viterbi decoding): {:.2f}% (known: {:.2f}%  novel: {:.2f}%)".format(correct_accuracy * 100,  known_accuracy * 100,  novel_accuracy * 100))

	# prob = 0
	# n = len(test_tags) - 1
	# curr = ('###', n)
	# while n > 0:
	# 	prob += mu[curr]
	# 	curr = (backpointers[curr], n - 1)
	# 	n = n - 1
	viterbi_perplexity = np.exp(-1 * (get_mu('###', len(test_tags)-1)) / len(test_tags))


	print 'Perplexity per Viterbi-tagged test word: {:.4f}'.format(viterbi_perplexity)
	print ("Tagging accuracy (posterior decoding): {:.2f}% (known: {:.2f}%  novel: {:.2f}%)".format(fb_correct_accuracy * 100,  fb_known_accuracy * 100,  fb_novel_accuracy * 100))
	'''
if __name__ == '__main__':
	main()
