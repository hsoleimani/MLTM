/*
 * opt.c
 *
 *  Created on: May 22, 2015
 *      Author: studadmin
 */

#include "opt.h"

//static gsl_vector thalpha_evaluate(void *instance, const gsl_vector *x,
//		gsl_vector *grad, const int n, const gsl_vector step);
double my_f (const gsl_vector *v, void *params);
void my_df (const gsl_vector *v, void *params, gsl_vector *df);
void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df);


void optimize_alpha(gsl_vector * x, void * data, int n, gsl_vector * x2){

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
	gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-3);

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

		status = gsl_multimin_test_gradient (s->gradient, 1e-3);

		if ((isnan(s->f)) || (isinf(s->f)))
			break;
		for (j = 0; j < n; j++){
			gsl_vector_set(x2, j, gsl_vector_get(s->x, j));
		}

	}while (status == GSL_CONTINUE && iter < 100);

	gsl_multimin_fdfminimizer_free (s);

}


double my_f (const gsl_vector *v, void *params)
{

	vblda_alphaopt * alphaopt=(vblda_alphaopt *) params;
	double f = 0.0;
	double alpha;

	alpha = exp(gsl_vector_get(v, 0));

	f = alphaopt->ndocs*(lgamma(alpha*alphaopt->ntopics)-alphaopt->ntopics*lgamma(alpha));
	f += (alpha-1)*alphaopt->ss;
	f = -f;

	return(f);
}



void my_df (const gsl_vector *v, void *params, gsl_vector *df)
{

	vblda_alphaopt * alphaopt=(vblda_alphaopt *) params;
	double alpha, grad;

	alpha = exp(gsl_vector_get(v, 0));

	grad = alphaopt->ndocs*alphaopt->ntopics*(gsl_sf_psi(alpha*alphaopt->ntopics)-gsl_sf_psi(alpha));
	grad += alphaopt->ss;
	gsl_vector_set(df, 0, -alpha*grad);

}


void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df)
{

	vblda_alphaopt * alphaopt=(vblda_alphaopt *) params;
	double alpha, grad;

	alpha = exp(gsl_vector_get(v, 0));

	*f = alphaopt->ndocs*(lgamma(alpha*alphaopt->ntopics)-alphaopt->ntopics*lgamma(alpha));
	*f += (alpha-1)*alphaopt->ss;
	grad = alphaopt->ndocs*alphaopt->ntopics*(gsl_sf_psi(alpha*alphaopt->ntopics)-gsl_sf_psi(alpha));
	grad += alphaopt->ss;
	gsl_vector_set(df, 0, -alpha*grad);
	*f = -(*f);

}


