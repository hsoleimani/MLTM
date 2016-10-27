
#ifndef cclda_H_
#define cclda_H_

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
	//double** MCzvec; //phi[i][j]
	int* y; //y[c] \in {0,1}
	double* HMCx;
	double* MCy; //y[c]
	int* x; //x[c] \in {1,...,Ls}
	double* MCx;
} sentence;


typedef struct
{
	sentence* sents; // sents[s]
	double* v; // for HMC
	int length;
	int* b; //b[c] binary label vector
	double* MCb; //only for test
} document;

typedef struct
{
	document* docs;
	int nterms;
	int ndocs;
} mltm_corpus;

typedef struct
{
	double phi;
	int s;
} hit_times;


typedef struct mltm_model
{
	int T; // # MC iterations
	int c; // # classes
	int m; // # topics
	int D; // # docs
	int n; // # terms
	int BurnIn;
	int P;
	double alpha; //prior on theta
	double nu; //prior on beta
	double MCalpha;
	double MCnu;
	double* psi0; //prior on beta psi[2]
	double** MCtheta; //theta[j][d]
	double** MCbeta; //beta[j][n]
	double** MCmu; //MCmu[j][c]
	double** MCpsi; //MCpsi[c][2]
	double** psi; //MCpsi[c][2]
} mltm_model;

typedef struct mltm_mcmc
{
	double n;
	int b; // block length for batch mean
	int a; // num of blocks for batch mean
	double zybar2;
	double zhat2;
	double zmeanybar2;
	double znum;
	double muybar2;
	double muhat2;
	double mumeanybar2;
	double mcse_z;
	double mcse_mu;
	double acceptrate;
	double acceptrate_psi;
	double acceptrate_alpha;
	double acceptrate_nu;
	double numYsampled; //number of nonzero Ys that we sample.
	double** t; //t[j][n]
	double** logt;
	double* tbar; //tbar[j]
	double* logtbar;
	double** m; //m[j][d]
	double** logm;
	double** K1; //K1[j][c]
	double** K2; //K2[j][c]
	double** logmu;
	double** log1mmu;
	double** YbarMCmu;
} mltm_mcmc;

typedef struct mltm_var
{
	hit_times* phi_vec;
	double* phi; //phi[j]
	double* gamma; //phi[i]
	double* temp1; //phi[j]
	double* temp1prime; //phi[j]
	double* temp2; //phi[j]
	double* temp2prime; //phi[j]
	unsigned int* z;
	int* yprime;
	int* xprime;
	double* yprob;
	int* numS;
	int* pick_sents;
	int num_pick_sents;
	int* xvec;
	int numxvec;
} mltm_var;



#endif /* cclda_H_ */
