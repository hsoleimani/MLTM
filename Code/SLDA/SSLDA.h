
#ifndef cclda_H_
#define cclda_H_

#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_vector.h>

#define NUM_INIT 20
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-50
#define PI 3.14159265359
#define min(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a < _b ? _a : _b; })


typedef struct
{
	int* words;
	int* counts;
	int length;
	int total;
	int* b; //b[c]
	double** phi; //phi[i][j]
	double* gamma; //gamma[j]
	double* sumphi;
	double sumgamma;
	double* xi; //xi[c]
} document;

typedef struct
{
	document* docs;
	int nterms;
	int ndocs;

	int m;
	int c;
	double* w;
	double* grad;
} sslda_corpus;


typedef struct sslda_model
{
	int T; // # iterations
	int c; // # classes
	int m; // # topics
	int D; // # docs
	int n; // # terms
	double logalpha;
	double alpha; //prior on theta
	double** beta; //beta[j][n]
	double** logbeta; //beta[j][n]
	double** w; //w[j][c]
} sslda_model;

typedef struct sslda_ss
{
	double** t; //t[j][n]
	double* sumt; //sumt[j]
	double alpha;
	double* oldphi;
	double* xi;
	double* oldxi;
	double* hi;
} sslda_ss;

typedef struct sslda_alphaopt
{
	int m;
	int d;
	double alpha;
	double ss;
} sslda_alphaopt;


#endif /* cclda_H_ */
