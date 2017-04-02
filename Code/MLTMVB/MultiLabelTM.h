
#ifndef cclda_H_
#define cclda_H_

#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
//#include <gsl/gsl_vector.h>


#define NUM_INIT 20
#define SEED_INIT_SMOOTH 1.0
#define EPS 1e-50
#define PI 3.14159265359
#define max(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a > _b ? _a : _b; })


typedef struct
{
	int* words;
	//int* counts;
	int length;

	int* samples; //samples[i]

	int* xsamples; //[c]
	double** x; //[c][ls]
	double* py; //[c]

	double* MCz; //phi[i]
	double* YbarMCz; // for batch means

	double** phi; //phi[i][j]

	int cumlen; //cumulative length of previous sentences
	int cumlen_topic; //cumulative length (*ntopics) of previous sentences
} sentence;


typedef struct
{
	sentence* sents; // sents[s]
	int length;
	int* b; //b[c] binary label vector
	double* pb; //[c]
	double* theta;
	double* logtheta;

	double* m; //[j]
	double mbar;

	double* logpb; //[c]
	double* logp1mb; //[c]

	double iteration;
	int rep;
	int a; // num of blocks for batch mean
	double zybar2;
	double zhat2;
	double zmeanybar2;
	double znum;
	double mcse_z;

	int* argmaxy; //[c]
	double* maxy; //[c]
} document;

typedef struct
{
	document* docs;
	int nterms;
	int ndocs;
	int max_length;
} mltm_corpus;


typedef struct mltm_model
{
	int b; // block length for batch mean
	int BurnIn;

	int T; // # MC iterations
	int c; // # classes
	int m; // # topics
	int D; // # docs
	int n; // # terms
	int L;
	double alpha;
	double nu;
	double alphahat;
	double nuhat;
	double KAPPA;
	double TAU;
	double s;
	int BATCHSIZE;
	double p;
	int lag;
	double lambda;
	int save_lag;
	double labeled_ratio;

	/*double adam_alpha;
	double adam_beta1;
	double adam_beta2;
	double nu_adam_alpha;
	double nu_adam_beta1;
	double nu_adam_beta2;
	double adam_eps;*/
	double rho0;

	//double** expElogbeta;
	double* Elogbeta; //[n+j]
	double** mu;// [c][j]
	//double** muhat; //[c][j]

	//double** MCw;// [c][j]
	//double** MCbeta;// [j][n]

	double psi1;
	double psi2;
	double psi1hat;
	double psi2hat;

} mltm_model;

typedef struct mltm_var
{
	double* gamma; //[j]
	double* phi; //[s+i+j]
	double* oldphi; //[j]
	double* Mphi;
	double* logm; //[j]
	//double* cdfphi; //[j]
	double* phibar; //[s+j]
	double* tempphi; //[j]
	double* sumphi; //[j]
	double sumgamma ;
	//double* Elogtheta; //[j]
	double* psigamma; //[j]

	double** mu; //[j][n]
	double* summu; //[j]

	double** gbar;//[j][n]
	double* hbar;//[j]
	double* rho; //[j]
	double* tau;//[j]

	double** mu_adam_m; //[j][n]
	double** mu_adam_v; //[j][n]

	//double* gradw; //[c+j]

	//int** grad_init;
	double* grad_temp;
	int* grad_temp_init;

	double** adam_m;
	double** adam_v;

	//optimizing alpha and nu
	long double adam_m_alphahat;
	long double adam_v_alphahat;
	long double adam_m_nuhat;
	long double adam_v_nuhat;
	double adam_m_psi1hat;
	double adam_v_psi1hat;
	double adam_m_psi2hat;
	double adam_v_psi2hat;

	double* px;
	double* py_ex;
	double* py_ex2;

	int* temp_argmaxy; //[i]
	double* temp_maxy; //[i]
	int** cj_temp_argmaxy; //[c][j]
	double** cj_temp_maxy; //[c][j]

} mltm_var;


typedef struct mltm_var_array
{
	mltm_var* var;
} mltm_var_array;

typedef struct mltm_ss
{
	double* beta;
	double* sumbeta;
	//double* theta;
	//double sumtheta;
	double* gradw_on; //[c+j]
	double* gradw_off; //[c+j]

	double* beta_thread;
	double* sumbeta_thread;
	//double* theta;
	//double sumtheta;
	//double* gradw_thread; //[c+j]
	double* gradw_on_thread; //[c+j]
	double* gradw_off_thread; //[c+j]
	//double* hessw_thread; //[c+j]

	double* alpha;
	double nu;
} mltm_ss;

/*typedef struct mltm_var
{
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
} mltm_var;*/



#endif /* cclda_H_ */
