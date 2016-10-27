
#ifndef cclda_H_
#define cclda_H_

//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_matrix.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>


#define NUM_INIT 20
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-50
#define PI 3.14159265359
#define min(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a < _b ? _a : _b; })


typedef struct
{
	int* words;
	int length;
	int* z; //phi[i] \in {1,...,M}
	double* MCz; //phi[i]
	double* YbarMCz; // for batch means
	double* MClblz; // label of each word based on topic-label association
} sentence;


typedef struct
{
	sentence* sents; // sents[s]
	int length;
	int* b; //b[c] binary label vector
	int* union_lbl; //b[c] binary label vector
} document;

typedef struct
{
	document* docs;
	int nterms;
	int ndocs;
} mltm_corpus;


typedef struct mltm_model
{
	int T; // # MC iterations
	int c; // # classes
	int m; // # topics
	int D; // # docs
	int n; // # terms
	int BurnIn;
	int general;
	double alpha; //prior on theta
	double nu; //prior on beta
	double MCalpha;
	double MCnu;
	double* psi0;
	double** MCtheta; //theta[j][d]
	double** MCbeta; //beta[j][n]
	int* lbl_of_tpc; //lbl_of_tpc[j] label of each topic
	int** tpcs_of_lbl; // topics of each label
	int m_per_c; //number of topics per class
} mltm_model;

typedef struct mltm_mcmc
{
	double n;
	int b; // block length for batch mean
	int a; // num of blocks for batch mean
	double acceptrate_alpha;
	double acceptrate_nu;
	double zybar2;
	double zhat2;
	double zmeanybar2;
	double znum;
	double mcse_z;
	double** t; //t[j][n]
	double** logt; //t[j][n]
	double* tbar; //tbar[j]
	double* logtbar; //tbar[j]
	double** m; //m[j][d]
	double** logm; //m[j][d]
} mltm_mcmc;

typedef struct mltm_var
{
	double* phi; //phi[j]
	double* gamma; //phi[i]
	double* temp1; //phi[j]
	double* temp1prime; //phi[j]
	double* temp2; //phi[j]
	double* temp2prime; //phi[j]
	unsigned int* z;
	int* yprime;
	double* yprob;
} mltm_var;



#endif /* cclda_H_ */
