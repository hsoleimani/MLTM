
#ifndef MAIN_H_
#define MAIN_H_

#include "MultiLabelTM.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
//#include <gsl/gsl_rng.h>
#include <math.h>
#include <omp.h>


#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
#define sign(a) ({a < 0 ? -1.0 : 1.0;})
const gsl_rng_type * T;
gsl_rng * rng;
int SENT;
double T_HMC;
int L0;
double TAU;
double CONVERGED;
int MAXITER;
int NUMC;
int MCSIZE;
int NUMINIT;
int BatchSize;
double Kappa;
int nthreads;

void find_max(document* doc, int chk, int s0, int i0, int j0, mltm_model* model);
int compare_function(const void *a,const void *b);
mltm_model* new_mltm_model(mltm_corpus* corpus, char* settings_file);
mltm_ss * new_mltm_ss(mltm_model* model);
mltm_var_array* new_mltm_var(mltm_corpus* corpus, mltm_model* model);


double testlda_doc_estep(document* doc, mltm_model* model, mltm_ss* ss,
		mltm_var* var, int d, int thread_id, int Lmax);

double doc_estep(document* doc, mltm_model* model, mltm_ss* ss, mltm_var* var,
		int d, int thread_id, int lbld, int iteration);

void train(char* dataset, char* lblfile, char* settings_file,
		char* start, char* dir, char* model_name);

int compare_function(const void *a,const void *b);

int main(int argc, char* argv[]);
void write_mltm_model(mltm_corpus * corpus, mltm_model * model, mltm_ss* ss, mltm_var* var, char * root, int chktest);
double compute_wrdlkh(document* doc, mltm_model* model, mltm_ss* ss, mltm_var* var, int d);
void loadtopics_initialize_model(mltm_corpus* corpus, mltm_model * model,
		mltm_ss* ss, mltm_var* var, char* model_root);
void load_model_continue(mltm_model* model, mltm_corpus* corpus, mltm_ss* ss,
		mltm_var* var, char* model_root);

void test_initialize(mltm_corpus* corpus, mltm_model* model, mltm_ss* ss, char* model_name);

void load_model(mltm_model* model, mltm_corpus* corpus, mltm_ss* ss, mltm_var* var, char* model_root);

mltm_corpus* read_data(const char* data_filename, int nclasses, int ndocs,
		int lblchck, const char* lbl_filename, int ntopics);

void test(char* dataset, char* settings_file, char* model_name, char* dir);
void random_initialize_model(mltm_corpus* corpus, mltm_model * model, mltm_ss* ss, mltm_var* var);
void corpus_initialize_model(mltm_corpus* corpus, mltm_model * model, mltm_ss* ss, mltm_var* var);
double log_sum(double log_a, double log_b);
double log_subtract(double log_a, double log_b);

double lda_doc_estep(document* doc, mltm_model* model, mltm_ss* ss, mltm_var* var,
		int d, int thread_id);
void random_permute(int size, int* vec);
#endif /* MAIN_H_ */
