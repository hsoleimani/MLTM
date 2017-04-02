
#ifndef cclda_H_
#define cclda_H_

#include <gsl/gsl_vector.h>
//#include <gsl/gsl_matrix.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>


#define NUM_INIT 20
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-50
#define PI 3.14159265359
#define maxm(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a > _b ? _a : _b; })


typedef struct
{
	int* words;
	int length;
	double* py; //Y[c]
} sentence;


typedef struct
{
	sentence* sents; // sents[s]
	int* argmaxs; //[c]
	double* maxs; //[c]
	int length;
	int* b; //b[c] binary label vector
} document;

typedef struct
{
	document* docs;
	int nterms;
	int ndocs;

	//params for optimizing w
	int c;
	double* grad;
	double* s;
	double* h;
} mllr_corpus;


typedef struct mllr_model
{
	int T; // # MC iterations
	int c; // # classes
	int D; // # docs
	int n; // # terms
	double** h; //w[c][n]
	double** s; //w[c][n]
	double* hTs; //hTs[c]
	//double** shat; //w[c][n]
} mllr_model;


typedef struct term_counts
{
	int* cnts;
	int num_occurrences;
} term_counts;


typedef struct mllr_wopt
	{
	int D; // # docs
	int n; // # terms
	double* grad; //w[c][n]
	document* docs;
} mllr_wopt;


#endif /* cclda_H_ */
