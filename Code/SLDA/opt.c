/*
 * opt.c
 *
 *  Created on: Oct 24, 2015
 *      Author: hossein
 */


#include "opt.h"

double my_f (const gsl_vector *v, void *params);
void my_df (const gsl_vector *v, void *params, gsl_vector *df);
void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df);
double alpha_f (const gsl_vector *v, void *params);
void alpha_df (const gsl_vector *v, void *params, gsl_vector *df);
void alpha_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df);


void optimize_w(gsl_vector * x, void * data, int n, gsl_vector * x2){

	size_t iter = 0;
	int status, j;

	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer *s;

	gsl_multimin_function_fdf my_func;

	my_func.n = n;
	my_func.f = my_f;
	my_func.df = my_df;
	my_func.fdf = my_fdf;
	my_func.params = data;

	T = gsl_multimin_fdfminimizer_conjugate_fr;
	s = gsl_multimin_fdfminimizer_alloc (T, n);

	gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.1, 0.01);

	do{
		iter++;
		status = gsl_multimin_fdfminimizer_iterate (s);
		//printf ("status = %s\n", gsl_strerror (status));
		if (status){
			if (iter == 1){
				for (j = 0; j < n; j++){
					gsl_vector_set(x2, j, gsl_vector_get(x, j));
				}
			}
			break;
		}

		status = gsl_multimin_test_gradient (s->gradient, 1e-2);

		if ((isnan(s->f)) || (isinf(s->f)))
			break;
		for (j = 0; j < n; j++){
			gsl_vector_set(x2, j, gsl_vector_get(s->x, j));
		}

	}while ((status == GSL_CONTINUE) && (iter < 20));

	gsl_multimin_fdfminimizer_free (s);
}



double my_f (const gsl_vector *v, void *params)
{

	sslda_corpus * corpus = (sslda_corpus *) params;
	document* doc;
	double f = 0.0;
	double nd, xi, cnt, h, h_i;
	int d, i, c, j;

	f = 0.0;
	for (j = 0; j < corpus->m; j++){
		corpus->w[j] = gsl_vector_get(v, j);
	}
	c = corpus->c;
	for (d = 0; d < corpus->ndocs; d++){
		if (corpus->docs[d].b[0] == -1) //skip unlabeled docs
			continue;

		doc = &(corpus->docs[d]);

		nd = (double) doc->total;
		if (doc->b[c] == 1){
			for (j = 0; j < corpus->m; j++){
				f += corpus->w[j]*doc->sumphi[j]/nd;
			}
		}
		xi = 1.0 + exp(doc->xi[c]);

		h = 0.0;
		for (i = 0; i < doc->length; i++){
			//n = doc->words[i];
			cnt = (double) doc->counts[i];

			//check = 0;
			h_i = 0.0;
			for (j = 0; j < corpus->m; j++){
				if (doc->phi[i][j] == 0)
					continue;
				h_i += doc->phi[i][j]*exp(cnt*corpus->w[j]/nd);
			}
			h += log(h_i);
		}

		f -= (1.0 + exp(h))/xi + log(xi) - 1.0;
	}


	f = -f;

	return(f);
}



void my_df (const gsl_vector *v, void *params, gsl_vector *df)
{

	sslda_corpus * corpus = (sslda_corpus *) params;
	double nd, xi, cnt, h, h_i;
	int d, i, c, j;
	document* doc;


	for (j = 0; j < corpus->m; j++){
		corpus->w[j] = gsl_vector_get(v, j);
		corpus->grad[j] = 0.0;
	}
	c = corpus->c;
	for (d = 0; d < corpus->ndocs; d++){
		if (corpus->docs[d].b[0] == -1) //skip unlabeled docs
			continue;

		doc = &(corpus->docs[d]);

		nd = (double) doc->total;
		if (doc->b[c] == 1){
			for (j = 0; j < corpus->m; j++){
				corpus->grad[j] += doc->sumphi[j]/nd;
			}
		}
		xi = 1.0 + exp(doc->xi[c]);

		h = 0.0;
		for (i = 0; i < doc->length; i++){
			//n = doc->words[i];
			cnt = (double) doc->counts[i];


			h_i = 0.0;
			for (j = 0; j < corpus->m; j++){
				if (doc->phi[i][j] == 0)
					continue;
				h_i += doc->phi[i][j]*exp(cnt*corpus->w[j]/nd);
			}
			h += log(h_i);
		}

		for (i = 0; i < doc->length; i++){
			//n = doc->words[i];
			cnt = (double) doc->counts[i];

			for (j = 0; j < corpus->m; j++){
				if (doc->phi[i][j] == 0)
					continue;
				h_i += doc->phi[i][j]*exp(cnt*corpus->w[j]/nd);
			}

			h_i = exp(h - log(h_i))*cnt/(xi*nd);
			for (j = 0; j < corpus->m; j++){
				corpus->grad[j] -= exp(cnt*corpus->w[j]/nd)*doc->phi[i][j]*h_i;
			}
		}

		//f -= (1.0 + exp(h))/xi + log(xi) - 1.0;
	}

	for (j = 0; j < corpus->m; j++){
		gsl_vector_set(df, j, -corpus->grad[j]);
	}


}


void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df)
{


	sslda_corpus * corpus = (sslda_corpus *) params;
	double nd, xi, cnt, h, h_i;
	int d, i, c, j;
	document* doc;


	for (j = 0; j < corpus->m; j++){
		corpus->w[j] = gsl_vector_get(v, j);
		corpus->grad[j] = 0.0;
	}
	(*f) = 0.0;
	c = corpus->c;
	for (d = 0; d < corpus->ndocs; d++){
		if (corpus->docs[d].b[0] == -1) //skip unlabeled docs
			continue;

		doc = &(corpus->docs[d]);

		nd = (double) doc->total;
		if (doc->b[c] == 1){
			for (j = 0; j < corpus->m; j++){
				corpus->grad[j] += doc->sumphi[j]/nd;
				(*f) += corpus->w[j]*doc->sumphi[j]/nd;
			}
		}
		xi = 1.0 + exp(doc->xi[c]);

		h = 0.0;
		for (i = 0; i < doc->length; i++){
			//n = doc->words[i];
			cnt = (double) doc->counts[i];

			h_i = 0.0;
			for (j = 0; j < corpus->m; j++){
				if (doc->phi[i][j] == 0)
					continue;
				h_i += doc->phi[i][j]*exp(cnt*corpus->w[j]/nd);
			}
			h += log(h_i);
		}

		for (i = 0; i < doc->length; i++){
			//n = doc->words[i];
			cnt = (double) doc->counts[i];

			h_i = 0.0;
			for (j = 0; j < corpus->m; j++){
				if (doc->phi[i][j] == 0)
					continue;
				h_i += doc->phi[i][j]*exp(cnt*corpus->w[j]/nd);
			}

			h_i = exp(h - log(h_i))*cnt/(xi*nd);
			for (j = 0; j < corpus->m; j++){
				corpus->grad[j] -= exp(cnt*corpus->w[j]/nd)*doc->phi[i][j]*h_i;
			}
		}

		(*f) -= (1.0 + exp(h))/xi + log(xi) - 1.0;
	}

	for (j = 0; j < corpus->m; j++){
		gsl_vector_set(df, j, -corpus->grad[j]);
	}

	(*f) = -(*f);
}


void optimize_alpha(gsl_vector * x, void * data, int n, gsl_vector * x2){

	size_t iter = 0;
	int status, j;

	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer *s;

	gsl_multimin_function_fdf my_func;

	my_func.n = n;
	my_func.f = alpha_f;
	my_func.df = alpha_df;
	my_func.fdf = alpha_fdf;
	my_func.params = data;

	T = gsl_multimin_fdfminimizer_conjugate_fr;
	s = gsl_multimin_fdfminimizer_alloc (T, n);

	gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-2);

	do
	{
		iter++;
		status = gsl_multimin_fdfminimizer_iterate (s);
		//printf ("status = %s\n", gsl_strerror (status));
		if (status){
			if (iter == 1){
				for (j = 0; j < n; j++){
					gsl_vector_set(x2, j, gsl_vector_get(x, j));
				}
			}
			break;
		}

		status = gsl_multimin_test_gradient (s->gradient, 1e-3);
		if ((isnan(s->f)) || (isinf(s->f)))
			break;
		for (j = 0; j < n; j++){
			gsl_vector_set(x2, j, gsl_vector_get(s->x, j));
		}

	}while (status == GSL_CONTINUE && iter < 100);

	gsl_multimin_fdfminimizer_free (s);
}



double alpha_f (const gsl_vector *v, void *params)
{

	sslda_alphaopt * aopt = (sslda_alphaopt *) params;
	double f = 0.0;
	double alpha;

	alpha = exp(gsl_vector_get(v, 0));

	f = aopt->d*(lgamma(alpha*aopt->m) - aopt->m*lgamma(alpha)) +
			(alpha-1)*aopt->ss;
	f = -f;

	return(f);
}



void alpha_df (const gsl_vector *v, void *params, gsl_vector *df)
{

	sslda_alphaopt * aopt = (sslda_alphaopt *) params;
	double grad = 0.0;
	double alpha;

	alpha = exp(gsl_vector_get(v, 0));

	grad = aopt->d*aopt->m*(gsl_sf_psi(alpha*aopt->m) - gsl_sf_psi(alpha)) +
			aopt->ss;

	gsl_vector_set(df, 0, -grad*alpha);

}


void alpha_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df)
{

	sslda_alphaopt * aopt = (sslda_alphaopt *) params;
	double grad = 0.0;
	double alpha;

	alpha = exp(gsl_vector_get(v, 0));

	(*f) = aopt->d*(lgamma(alpha*aopt->m) - aopt->m*lgamma(alpha)) +
			(alpha-1)*aopt->ss;
	grad = aopt->d*aopt->m*(gsl_sf_psi(alpha*aopt->m) - gsl_sf_psi(alpha)) +
			aopt->ss;

	gsl_vector_set(df, 0, -grad*alpha);

	(*f) = -(*f);
}

double log_sum(double log_a, double log_b)
{
	double v;

	if (log_a < log_b){
		v = log_b+log(1 + exp(log_a-log_b));
	}else{
		v = log_a+log(1 + exp(log_b-log_a));
	}
	return(v);
}
