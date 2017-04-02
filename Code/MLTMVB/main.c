#include "main.h"



int main(int argc, char* argv[])
{

	char task[40];
	char dir[400];
	char corpus_file[400];
	char label_file[400];
	char settings_file[400];
	char model_name[400];
	char init[400];
	long int seed;
	int i;

	//omp_set_num_threads(1);
	nthreads = omp_get_max_threads();

	seed = atoi(argv[1]);
	rng = (gsl_rng *) malloc (sizeof (gsl_rng)*nthreads);
	for (i = 0; i < nthreads; i++){
		gsl_rng_env_setup();

		T = gsl_rng_default;
		rng[i] = *(gsl_rng_alloc (T));
		gsl_rng_set (&(rng[i]), seed + i*10203);

		printf("SEED = %ld\n", seed + i*10203);
	}

	MAXITER = 7000;
	L0 = 100;
	CONVERGED = 1e-4;
	NUMINIT = 10;

	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);

	if (argc > 1)
	{
		if (strcmp(task, "train")==0)
		{
			strcpy(label_file,argv[4]);
			strcpy(settings_file,argv[5]);
			strcpy(init,argv[6]);
			strcpy(dir,argv[7]);
			if ((strcmp(init, "load") == 0) ||
					(strcmp(init, "loadtopics") == 0) ||
					(strcmp(init, "continue") == 0))
				strcpy(model_name, argv[8]);
			train(corpus_file, label_file, settings_file, init, dir, model_name);

			//for (i = 0; i < nthreads; i++){
			//	gsl_rng_free (&(rng[i]));
			//}
			gsl_rng_free(rng);
			return(0);
		}
		if (strcmp(task, "test")==0)
		{
			strcpy(settings_file,argv[4]);
			strcpy(model_name,argv[5]);
			strcpy(dir,argv[6]);
			test(corpus_file, settings_file, model_name, dir);
			gsl_rng_free (rng);
			return(0);
		}
	    }
	return(0);
}


void train(char* dataset, char* lblfile, char* settings_file,
		char* start, char* dir, char* model_name)
{
	FILE* lhood_fptr;
	FILE* fp;
	char string[100];
	char filename[100];
	int ntopics, nclasses, ndocs, dd;
	double lkh=0, doclkh, temp, prevlkh, conv=-100;//, htemp, gbarnorm;
	double rho;//, summu;
	//double doclkh, wrdlkh;
	//double normsum, temp, psi_sigma, alpha_sigma, nu_sigma;
	int iteration;
	int d, n, j, c;
	long int t0;
	float y0;
	double ccr, doc_ccr, temp2, nccr, prev_ccr;

	mltm_corpus* corpus;
	mltm_model *model = NULL;
	mltm_ss *ss = NULL;
	//mltm_phiopt *phiopt = NULL;
	//mltm_mc* mc = NULL;
	mltm_var *var = NULL;
	mltm_var_array *var_array = NULL;

	time_t t1,t2;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &ndocs);
	//fscanf(fp, "N %d\n", &d);
	//fscanf(fp, "T %d\n", &MAXITER);
	//fscanf(fp, "L %d\n", &L0);
	//fscanf(fp, "alpha %lf\n", &L0);
	fclose(fp);

	corpus = read_data(dataset, nclasses, ndocs, 1, lblfile, ntopics);
	model = new_mltm_model(corpus, settings_file);
	ss = new_mltm_ss(model);


	var_array = new_mltm_var(corpus, model);
	var = &(var_array->var[0]);

	corpus->nterms = model->n;


	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	if (strcmp(start, "continue")==0){
		//load model
		load_model_continue(model, corpus, ss, var, model_name);

		// read t0, iteration
		// set up the log likelihood log file
		sprintf(string, "%s/likelihood.dat", dir);
		lhood_fptr = fopen(string, "r");
		while ((fscanf(lhood_fptr, "%d ", &iteration) != EOF)){
			fscanf(lhood_fptr, "%e %lf %ld %lf", &y0, &rho, &t0,&prev_ccr);
		}
		fclose(lhood_fptr);
		iteration += 1;
		printf("picked up from iteration %d, time %ld\n", iteration, t0);

		lhood_fptr = fopen(string, "a");
		/*for (c = 0; c < model->c; c++){
			for (j = 0; j < model->m; j++){
				var->adam_m[c][j] = 0.0;
				var->adam_v[c][j] = 0.0;
			}
		}*/
	}
	else{
		prev_ccr = 0.0;
		if (strcmp(start, "random")==0){
			printf("random\n");
			random_initialize_model(corpus, model, ss, var);

		}
		else if (strcmp(start, "seeded")==0){
			printf("seeded\n");
			corpus_initialize_model(corpus, model, ss, var);

		}
		else if (strcmp(start, "loadtopics")==0){
			printf("seeded\n");
			loadtopics_initialize_model(corpus, model, ss, var, model_name);

		}
		else if (strcmp(start, "load")==0){
			printf("load\n");

			load_model(model, corpus, ss, var, model_name);


		}

		t0 = 0;
		for (j = 0; j < model->m; j++){
			var->tau[j] = 2.0;//(double)model->BATCHSIZE;
		}
		// set up the log likelihood log file
		sprintf(string, "%s/likelihood.dat", dir);
		lhood_fptr = fopen(string, "w");
		iteration = 1;
	}

	//init ss
	for (j = 0; j < model->m; j++){
		var->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			var->summu[j] += var->mu[j][n];
		}
		temp = gsl_sf_psi(var->summu[j]);
		for (n = 0; n < model->n; n++){
			//y = var->mu[j][n]/var->summu[j];
			//if (y == 0) y = 1e-50;
			//model->expElogbeta[j][n] = y;
			model->Elogbeta[n*ntopics+j] = gsl_sf_psi(var->mu[j][n])-temp;
		}
	}

	//zero init tss
	int nj_ind, thread_id;
	for (thread_id = 0; thread_id < nthreads; thread_id++){
		for (j = 0; j < model->m; j++){
			ss->sumbeta_thread[thread_id*ntopics + j] = 0.0;
			for (n = 0; n < model->n; n++){
				nj_ind = thread_id*ntopics*model->n + n*ntopics;
				ss->beta_thread[nj_ind + j] = 0.0;
			}
			for (c = 0; c < model->c; c++){
				//ss->gradw_thread[thread_id*ntopics*model->c + c*ntopics+j] = 0.0;
				ss->gradw_on_thread[thread_id*ntopics*model->c + c*ntopics+j] = 0.0;
				ss->gradw_off_thread[thread_id*ntopics*model->c + c*ntopics+j] = 0.0;
			}
		}
	}


	//sprintf(filename, "%s/%03d", dir, 0);
	//printf("%s\n",filename);
	//write_mltm_model(corpus, model, ss, var, filename, 0);

	time(&t1);
	sprintf(filename, "%s/likelihood.dat", dir);

	//wrdlkh = 0.0;thr
	//sentence* sent;
	int cj_ind, temp_ind;
	prevlkh = -1e100;
	int* selected_docs = malloc(sizeof(int)*model->BATCHSIZE);

	//double grad_alpha, grad_nu, adam_beta1, adam_beta2;
	//double nlabeled;
	/*double** phi = malloc(sizeof(double*)*nthreads);
	double** oldphi = malloc(sizeof(double*)*nthreads);
	for (dd = 0; dd < nthreads; dd++){
		phi[dd] = malloc(sizeof(double)*ntopics);
		oldphi[dd] = malloc(sizeof(double)*ntopics);
		for (j = 0; j < ntopics; j++){
			phi[dd][j] = 0.0;
			oldphi[dd][j] = 0.0;
		}
	}*/

	int* labeled_docs = malloc(sizeof(int)*corpus->ndocs);
	int* unlabeled_docs = malloc(sizeof(int)*corpus->ndocs);
	int nlabeled = 0;
	int nunlabeled = 0;
	for (d = 0; d < corpus->ndocs; d++){
		if (corpus->docs[d].b[0] == -1){
			unlabeled_docs[nunlabeled] = d;
			nunlabeled += 1;
		}else{
			labeled_docs[nlabeled] = d;
			nlabeled += 1;
		}
	}
	//estimate beta priors
	/*double psi_ex = 0.0, psi_ex2, psi_num;
	double psi_var, psi_mean;
	for (c = 0; c < model->c; c++){
		temp = 0.0;
		for (dd = 0; dd < nlabeled; dd++){
			d = labeled_docs[dd];
			temp += corpus->docs[d].b[c];
		}
		temp /= (double)nlabeled;
		psi_ex += temp;
		psi_ex2 += temp*temp;
	}
	psi_num = (double)model->c;
	psi_mean = psi_ex/psi_num;
	psi_var = psi_ex2/psi_num - psi_mean*psi_mean;
	psi_var *= psi_num/(psi_num-1.0);
	if (psi_var < psi_mean*(1-psi_mean)){
		temp = psi_mean*(1-psi_mean)/psi_var-1.0;
		model->psi1 = psi_mean*temp;
		model->psi2 = (1-psi_mean)*temp;
	}else{
		model->psi1 = 2.0;
		model->psi2 = 2.0;
	}
	//overwrite these
	model->psi1 = 1.0;
	model->psi2 = 1.0;
	printf("psi: %lf %lf | %lf %lf %lf\n", model->psi1, model->psi2, psi_var, psi_mean*(1-psi_mean), psi_mean);*/
	printf("psi: %lf %lf\n", model->psi1, model->psi2);
	model->psi1hat = log(model->psi1);
	model->psi2hat = log(model->psi2);

	model->labeled_ratio = (double)nlabeled/((double)corpus->ndocs);
	int batch_lbld, batch_unlbld;
	batch_lbld = (int)floor((model->labeled_ratio)*model->BATCHSIZE);
	if (batch_lbld < 0.05*model->BATCHSIZE)
		batch_lbld = (int)floor(0.05*model->BATCHSIZE);
	if (batch_lbld > nlabeled)
		batch_lbld = nlabeled;

	batch_unlbld = model->BATCHSIZE - batch_lbld;
	int* random_order = malloc(sizeof(int)*model->BATCHSIZE);
	int* lbld_random_order = malloc(sizeof(int)*nlabeled);
	int* ulbld_random_order = malloc(sizeof(int)*nunlabeled);

	printf("******** %d %d, %d %d\n", nlabeled, nunlabeled, batch_lbld,batch_unlbld);
	ccr = 0.0;
	//int tcnt;
	int ready_to_break = 0;
	int iter_batchsize, d3, rep_chk;
	//model->BurnIn = 10;
	model->b = 30;
	double mcstderr, avg_mcstderr, doc_rho;
	int nmcstderr, wskip;
	double* max_mcstderr = malloc(sizeof(double)*nthreads);
	//double grad_alpha, grad_nu, adam_beta1, adam_beta2;
	//double iter_alpha = 0;
	if (model->BATCHSIZE == corpus->ndocs){
		for (d = 0; d < corpus->ndocs; d++){
			random_order[d] = d;
			selected_docs[d] = d;
			corpus->docs[d].rep = 1;
		}
	}

	double adam_beta1, adam_beta2, adam_alpha, grad_nu, alpha_t;
	double grad_alpha, grad_psi1, grad_psi2;

	do{

		if (model->BATCHSIZE == corpus->ndocs){
			iter_batchsize = model->BATCHSIZE;
			batch_lbld = nlabeled;
			batch_unlbld = 0;
		}
		else{
			//no replacement
			if (0){
				random_permute(nlabeled, lbld_random_order);
				random_permute(nunlabeled, ulbld_random_order);
				iter_batchsize = 0;
				for (d3 = 0; d3 < batch_lbld; d3++){
					d = lbld_random_order[d3];
					selected_docs[iter_batchsize] = d;
					iter_batchsize += 1;
					corpus->docs[d].rep = 1;
				}
				for (d3 = 0; d3 < batch_unlbld; d3++){
					d = ulbld_random_order[d3];
					selected_docs[iter_batchsize] = d;
					iter_batchsize += 1;
					corpus->docs[d].rep = 1;
				}
				if (iter_batchsize != model->BATCHSIZE){
					printf("ooooops\n");
				}
				//shuffle selected docs
				random_permute(iter_batchsize, random_order);

			}else{
				iter_batchsize = 0;
				//batch_lbld = 0;
				//for (dd = 0; dd < model->BATCHSIZE; dd++){
				dd = 0;
				do{

					// choose a document; separate labeled and unlabeled selection
					if (dd < batch_lbld){

						if (batch_lbld < nlabeled)
							d = labeled_docs[(int)floor(gsl_rng_uniform(&(rng[0])) * (nlabeled))];
						else
							d = labeled_docs[dd];
					}
					else{
						d = unlabeled_docs[(int)floor(gsl_rng_uniform(&(rng[0])) * (nunlabeled))];
					}
					//choose a document; uniformly over all samples
					//d = floor(gsl_rng_uniform(&(rng[0])) * (corpus->ndocs));

					//check if d was selected before
					rep_chk = 0;
					for (d3 = 0; d3 < iter_batchsize; d3++){
						if (selected_docs[d3] == d){
							rep_chk = 1;
							break;
							//corpus->docs[d].rep += 1;
							//break;
						}
					}

					if (rep_chk == 0){
						selected_docs[iter_batchsize] = d;
						iter_batchsize += 1;
						corpus->docs[d].rep = 1;
						dd += 1;
						//printf("%d-%d-%d, ",iter_batchsize,d,dd);
					}


				}while(dd < model->BATCHSIZE);
				//printf("(%d)\n", model->BATCHSIZE);
				//shuffle selected docs
				random_permute(iter_batchsize, random_order);
			}

			//*********************************
			//with replacement
			if (0){
				iter_batchsize = 0;
				//batch_lbld = 0;
				for (dd = 0; dd < model->BATCHSIZE; dd++){

					// choose a document; separate labeled and unlabeled selection
					if (dd < batch_lbld){

						if (batch_lbld < nlabeled)
							d = labeled_docs[(int)floor(gsl_rng_uniform(&(rng[0])) * (nlabeled))];
						else
							d = labeled_docs[dd];
					}
					else{
						d = unlabeled_docs[(int)floor(gsl_rng_uniform(&(rng[0])) * (nunlabeled))];
					}
					//choose a document; uniformly over all samples
					//d = floor(gsl_rng_uniform(&(rng[0])) * (corpus->ndocs));

					//check if d was selected before
					rep_chk = 0;
					for (d3 = 0; d3 < iter_batchsize; d3++){
						if (selected_docs[d3] == d){
							rep_chk = 1;
							corpus->docs[d].rep += 1;
							break;
						}
					}

					if (rep_chk == 0){
						selected_docs[iter_batchsize] = d;
						iter_batchsize += 1;
						corpus->docs[d].rep = 1;
						//printf("%d-%d-%d, ",iter_batchsize,d,dd);
					}


				}
				//printf("(%d)\n", model->BATCHSIZE);
				//shuffle selected docs
				random_permute(iter_batchsize, random_order);
			}
		}


		rho = 0.0;
		# pragma omp parallel shared (rng,corpus, model, ss, var_array, random_order, selected_docs, iter_batchsize)\
		 private (d, thread_id, dd, doc_rho)  reduction (+:rho)
		{
			thread_id = omp_get_thread_num();
			# pragma omp for schedule(dynamic) 

			for (dd = 0; dd < iter_batchsize; dd++){
				d = selected_docs[random_order[dd]];
				if (corpus->docs[d].b[0] == -1)
					doc_rho = doc_estep(&(corpus->docs[d]), model, ss, &(var_array->var[thread_id]), d, thread_id,
							0, iteration);
				else{
					doc_rho = doc_estep(&(corpus->docs[d]), model, ss, &(var_array->var[thread_id]), d, thread_id,
						1, iteration);
				}
				rho += doc_rho;
			}

		}
		rho /= (double)model->BATCHSIZE;
		printf("%lf\n", rho);


		// update lambda

		//Adaptive rho
		/*rho = 0.0;
		htemp = 0.0;
		gbarnorm = 0.0;
		for (j = 0; j < model->m; j++){
			ss->sumbeta[j] = 0.0;
			for (thread_id = 0; thread_id < nthreads; thread_id++){
				temp_ind = thread_id*ntopics + j;
				ss->sumbeta[j] += ss->sumbeta_thread[temp_ind];
				ss->sumbeta_thread[temp_ind] = 0.0;
			}
			for (n = 0; n < model->n; n++){
				nj_ind = n*ntopics;
				ss->beta[nj_ind+j] = 0.0;
				for (thread_id = 0; thread_id < nthreads; thread_id++){
					temp_ind = thread_id*ntopics*model->n + nj_ind+j;
					ss->beta[nj_ind+j] += ss->beta_thread[temp_ind];
					ss->beta_thread[temp_ind] = 0.0;
				}
				temp = model->nu+((double)corpus->ndocs*ss->beta[nj_ind+j])/((double)model->BATCHSIZE)-var->mu[j][n];
				var->gbar[j][n] = var->gbar[j][n]*(1-1.0/var->tau[j])+temp/var->tau[j];
				htemp += temp*temp;
				gbarnorm += pow(var->gbar[j][n],2.0);
			}
		}
		//same learning rate for all topics
		for (j = 0; j < model->m; j++){
			var->hbar[j] = var->hbar[j]*(1-1.0/var->tau[j])+htemp/var->tau[j];
			var->rho[j] = gbarnorm/var->hbar[j];
			var->tau[j] = var->tau[j]*(1-var->rho[j])+1;
		}

		rho = 0.0;
		ss->nu = 0.0;
		for (j = 0; j < model->m; j++){
			rho += var->rho[j];
			var->summu[j] = (1-var->rho[j])*var->summu[j] + var->rho[j]*((double)model->n*model->nu)
							+ var->rho[j]*((double)corpus->ndocs*ss->sumbeta[j])/((double)model->BATCHSIZE);

			temp = gsl_sf_psi(var->summu[j]);

			for (n = 0; n < model->n; n++){
				nj_ind = n*ntopics;
				var->mu[j][n] = (1-var->rho[j])*var->mu[j][n] + var->rho[j]*(model->nu)
						+ var->rho[j]*((double)corpus->ndocs*ss->beta[nj_ind+j])/((double)model->BATCHSIZE);

				ss->beta[nj_ind+j] = 0.0;

				temp2 = (gsl_sf_psi(var->mu[j][n])-temp);
				model->Elogbeta[n*ntopics+j] = temp2;

				ss->nu += temp2;
				//model->expElogbeta[j][n] = exp(model->Elogbeta[n*ntopics+j]);
			}
			ss->sumbeta[j] = 0.0;

		}
		rho /= (double)model->m;
		printf("%d %d %lf %lf\n", iteration, j, var->rho[0], rho);*/

		// manuall learning rate
		//rho = model->adam_alpha*pow(1.0/(iteration+model->TAU), model->KAPPA);
		# pragma omp parallel default (shared) private (j, thread_id, n, temp, nj_ind, temp_ind)
		{
			# pragma omp for schedule(dynamic)
			//ss->nu = 0.0;
			for (j = 0; j < model->m; j++){

				ss->sumbeta[j] = 0.0;
				for (thread_id = 0; thread_id < nthreads; thread_id++){
					temp_ind = thread_id*ntopics + j;
					ss->sumbeta[j] += ss->sumbeta_thread[temp_ind];
					ss->sumbeta_thread[temp_ind] = 0.0;
				}

				var->summu[j] = (1-rho)*var->summu[j] + rho*((double)model->n*model->nu)
								+ rho*((double)corpus->ndocs*ss->sumbeta[j])/((double)model->BATCHSIZE);

				temp = gsl_sf_psi(var->summu[j]);
				for (n = 0; n < model->n; n++){

					nj_ind = n*ntopics;
					ss->beta[nj_ind+j] = 0.0;
					for (thread_id = 0; thread_id < nthreads; thread_id++){
						temp_ind = thread_id*ntopics*model->n + nj_ind+j;
						ss->beta[nj_ind+j] += ss->beta_thread[temp_ind];
						ss->beta_thread[temp_ind] = 0.0;
					}

					var->mu[j][n] = (1-rho)*var->mu[j][n] + rho*(model->nu)
							+ rho*((double)corpus->ndocs*ss->beta[nj_ind+j])/((double)model->BATCHSIZE);

					//temp2 = (gsl_sf_psi(var->mu[j][n])-temp);
					model->Elogbeta[n*ntopics+j] = (gsl_sf_psi(var->mu[j][n])-temp);//temp2;

					//ss->nu += temp2;

				}
				ss->sumbeta[j] = 0.0;
			}

		}

		//update w
		wskip = 0;
		grad_psi1 = 0.0;
		grad_psi2 = 0.0;
		if (iteration > wskip){
			rho *= model->rho0;
			for (c = 0; c < model->c; c++){
				cj_ind = c*ntopics;
				for (j = 0; j < model->m; j++){
					ss->gradw_on[cj_ind+j] = 0.0;
					ss->gradw_off[cj_ind+j] = 0.0;
					for (thread_id = 0; thread_id < nthreads; thread_id++){
						temp_ind = thread_id*ntopics*model->c + cj_ind + j;
						ss->gradw_on[cj_ind+j] += ss->gradw_on_thread[temp_ind];
						ss->gradw_off[cj_ind+j] += ss->gradw_off_thread[temp_ind];

						ss->gradw_on_thread[temp_ind] = 0.0;
						ss->gradw_off_thread[temp_ind] = 0.0;

					}

					if (ss->gradw_on[cj_ind+j] +  ss->gradw_off[cj_ind+j]== 0)
						continue;

					temp2 = (double)batch_lbld/(double)nlabeled;
					temp = (temp2*(model->psi1-1.0)+ss->gradw_on[cj_ind+j])/
							(temp2*(model->psi1+model->psi2-2.0)+ss->gradw_on[cj_ind+j] +  ss->gradw_off[cj_ind+j]);

					//if (iteration > model->BurnIn){
						model->mu[c][j] = (1-rho)*model->mu[c][j] +	rho*temp;
					//}else{
					//	model->mu[c][j] = temp;
					//}

					/*if (iteration > model->BurnIn){
						model->MCw[c][j] = (((double)iteration-1.0-model->BurnIn)*model->MCw[c][j]+
										model->mu[c][j])/((double)iteration-model->BurnIn);
					}else{
						model->MCw[c][j] = model->mu[c][j];
					}*/

					if (model->mu[c][j] >= 0.9999){
						model->mu[c][j] = 0.9999;
						//model->muhat[c][j] = log(0.9999);
					}
					if (model->mu[c][j] <= 0.0001){
						model->mu[c][j] = 0.0001;
						//model->muhat[c][j] = log(0.0001);
					}
					grad_psi1 += log(model->mu[c][j]);
					grad_psi2 += log(1-model->mu[c][j]);
				}
			}
		}
		else{
			for (c = 0; c < model->c; c++){
				cj_ind = c*ntopics;
				for (j = 0; j < model->m; j++){
					for (thread_id = 0; thread_id < nthreads; thread_id++){
						temp_ind = thread_id*ntopics*model->c + cj_ind+j;
						ss->gradw_on_thread[temp_ind] = 0.0;
						ss->gradw_off_thread[temp_ind] = 0.0;
					}
				}
			}
		}

		wskip = 1000;
		if (iteration > wskip){
			adam_beta1 = 0.9;
			adam_beta2 = 0.999;
			adam_alpha = 0.001;
			//update psi
			temp = gsl_sf_psi(model->psi1+model->psi2);
			grad_psi1 += 1.0/model->psi1 - 1.0/0.5 + model->m*model->c*(temp-gsl_sf_psi(model->psi1));
			grad_psi2 += 1.0/model->psi2 - 1.0/0.5 + model->m*model->c*(temp-gsl_sf_psi(model->psi2));
			grad_psi1 *= model->psi1;
			grad_psi2 *= model->psi2;

			/*var->adam_m_psi1hat = adam_beta1*var->adam_m_psi1hat+(1-adam_beta1)*grad_psi1;
			var->adam_v_psi1hat = adam_beta2*var->adam_v_psi1hat+(1-adam_beta2)*pow(grad_psi1,2.0);
			var->adam_m_psi2hat = adam_beta1*var->adam_m_psi2hat+(1-adam_beta1)*grad_psi2;
			var->adam_v_psi2hat = adam_beta2*var->adam_v_psi2hat+(1-adam_beta2)*pow(grad_psi2,2.0);
			alpha_t = adam_alpha*sqrt(1-pow(adam_beta2, (double)iteration-wskip))/
					(1-pow(adam_beta1, (double)iteration-wskip));*/

			var->adam_m_psi1hat += grad_psi1*grad_psi1;
			var->adam_m_psi2hat += grad_psi2*grad_psi2;
			alpha_t = adam_alpha;
			//model->psi1hat += alpha_t*var->adam_m_psi1hat/(sqrt(var->adam_v_psi1hat)+1e-8);
			model->psi1hat += alpha_t*grad_psi1/(sqrt(var->adam_m_psi1hat)+1e-8);
			model->psi1 = exp(model->psi1hat);
			//model->psi2hat += alpha_t*var->adam_m_psi2hat/(sqrt(var->adam_v_psi2hat)+1e-8);
			model->psi2hat += alpha_t*grad_psi2/(sqrt(var->adam_m_psi2hat)+1e-8);
			model->psi2 = exp(model->psi2hat);
			if (model->psi1 <= 1e-4){
				model->psi1 = 1e-4;
				model->psi1hat = log(1e-4);
				var->adam_m_psi1hat = 0;
				var->adam_v_psi1hat = 0;
			}
			if (model->psi2 <= 1e-4){
				model->psi2 = 1e-4;
				model->psi2hat = log(1e-4);
				var->adam_m_psi2hat = 0;
				var->adam_v_psi2hat = 0;
			}
			if (model->psi1 >= 2){
				model->psi1 = 2;
				model->psi1hat = log(2);
				var->adam_m_psi1hat = 0;
				var->adam_v_psi1hat = 0;
			}
			if (model->psi2 >= 2){
				model->psi2 = 2;
				model->psi2hat = log(2);
				var->adam_m_psi2hat = 0;
				var->adam_v_psi2hat = 0;
			}
			//printf(">>> %lf, %lf - %lf, %lf\n", model->psi1, model->psi1hat, model->psi2, model->psi2hat);
			//printf("> %lf, %lf - %lf, %lf\n", grad_psi1, var->adam_m_psi1hat, grad_psi2, var->adam_m_psi2hat);
			//printf(">> %lf %lf\n", model->psi1, model->psi2);
			//if (model->psi1 <= 1e-4){
			//	model->nu = 1e-4;
			//	model->nuhat = log(1e-4);
			//}
			//if (model->nu >= 1){
			//	model->nu = 1;
			//	model->nuhat = log(1);
			//}

			//update nu
			/*grad_nu = model->m*model->n*(gsl_sf_psi(model->nu*model->n)-gsl_sf_psi(model->nu))+
					ss->nu;
			grad_nu += (1.0/model->nu - 1.0/0.5);
			grad_nu *= model->nu;
			var->adam_m_nuhat = adam_beta1*var->adam_m_nuhat+(1-adam_beta1)*grad_nu;
			var->adam_v_nuhat = adam_beta2*var->adam_v_nuhat+(1-adam_beta2)*pow(grad_nu,2.0);
			alpha_t = adam_alpha*sqrt(1-pow(adam_beta2, (double)iteration-wskip))/
					(1-pow(adam_beta1, (double)iteration-wskip));

			model->nuhat += alpha_t*var->adam_m_nuhat/(sqrt(var->adam_v_nuhat)+1e-8);
			model->nu = exp(model->nuhat);
			if (model->nu <= 1e-4){
				model->nu = 1e-4;
				model->nuhat = log(1e-4);
			}
			if (model->nu >= 1){
				model->nu = 1;
				model->nuhat = log(1);
			}
			ss->nu = 0.0;
			//update alpha
			grad_alpha = 0.0;
			for (thread_id = 0; thread_id < nthreads; thread_id++){
				grad_alpha += ss->alpha[thread_id];
				ss->alpha[thread_id] = 0.0;
			}
			grad_alpha += ((double)corpus->ndocs)*(1.0/model->alpha - 1.0/0.5)/((double)iter_batchsize);
			grad_alpha *= model->alpha;
			var->adam_m_alphahat = adam_beta1*var->adam_m_alphahat+(1-adam_beta1)*grad_alpha;
			var->adam_v_alphahat = adam_beta2*var->adam_v_alphahat+(1-adam_beta2)*pow(grad_alpha,2.0);
			alpha_t = adam_alpha*sqrt(1-pow(adam_beta2, (double)iteration-wskip))/
					(1-pow(adam_beta1, (double)iteration-wskip));
			model->alphahat += alpha_t*var->adam_m_alphahat/(sqrt(var->adam_v_alphahat)+1e-8);
			model->alpha = exp(model->alphahat);
			if (model->alpha <= 1e-4){
				model->alpha = 1e-4;
				model->alphahat = log(1e-4);
				var->adam_m_alphahat = 0.0;
				var->adam_v_alphahat = 0.0;
			}
			if (model->alpha >= 0.5){
				model->alpha = 0.5;
				model->alphahat = log(0.5);
				var->adam_m_alphahat = 0.0;
				var->adam_v_alphahat = 0.0;
			}*/
		}

		for (thread_id = 0; thread_id < nthreads; thread_id++){
			ss->alpha[thread_id] = 0.0;
		}



		if ((iteration % model->lag == 0) && (iteration > 0)){

			lkh = 0;
			ccr = 0.0;
			nccr = 0.0;
			mcstderr = -1e4;
			for (thread_id = 0; thread_id < nthreads; thread_id ++){
				max_mcstderr[thread_id] = -1e100;
			}
			nmcstderr = 0;
			avg_mcstderr = 0.0;

			# pragma omp parallel shared (corpus, model, ss, var_array) \
			private (d, thread_id, c, doc_ccr) \
			reduction ( + : lkh, ccr, nccr, nmcstderr, avg_mcstderr)
			{
				thread_id = omp_get_thread_num();
				# pragma omp for schedule(dynamic)
				for (d = 0; d < corpus->ndocs; d++){

					lkh += testlda_doc_estep(&(corpus->docs[d]), model,
							ss, &(var_array->var[thread_id]), d, thread_id, 50);

					if ((corpus->docs[d].mcse_z > 0)){
						if (corpus->docs[d].mcse_z > max_mcstderr[thread_id])
							max_mcstderr[thread_id] = corpus->docs[d].mcse_z;
						nmcstderr += 1;
						avg_mcstderr += corpus->docs[d].mcse_z;
					}

					doc_ccr = 0.0;
					if (corpus->docs[d].b[0] != -1){
						for (c = 0; c < model->c; c++){
							if ((corpus->docs[d].b[c] == 0) && (corpus->docs[d].pb[c] < 0.5))
								doc_ccr += 1.0;
							else if  ((corpus->docs[d].b[c] == 1) && (corpus->docs[d].pb[c] >= 0.5))
								doc_ccr += 1.0;
						}
						doc_ccr /= (double) model->c;
						ccr += doc_ccr;	
						nccr += 1.0;
					}
				}
			}
			ccr /= nccr;//(double)corpus->ndocs;

			avg_mcstderr /= nmcstderr;
			for (thread_id = 0; thread_id < nthreads; thread_id ++){
				if (max_mcstderr[thread_id] > mcstderr)
					mcstderr = max_mcstderr[thread_id];
			}

			lkh += model->m*(lgamma(model->nu*model->n)-model->n*lgamma(model->nu));
			lkh += corpus->ndocs*(lgamma(model->alpha*model->m)-model->m*lgamma(model->alpha));
			for (j = 0; j < model->m; j++){
				for (n = 0; n < model->n; n++){
					lkh += lgamma(var->mu[j][n]);
				}
				lkh -= lgamma(var->summu[j]);
				for (c = 0; c < model->c; c++){
					lkh += ((model->psi1-1.0)*log(model->mu[c][j])+
							(model->psi2-1.0)*log(1-model->mu[c][j]));///((double)nlabeled); //regularization
				}
			}
			conv = fabs(prevlkh - lkh)/fabs(prevlkh);
			prevlkh = lkh;
			if (avg_mcstderr < 0.05)
				ready_to_break = 1;

			prev_ccr = ccr;
		}


		if (iteration % model->save_lag == 0){
			sprintf(filename, "%s/%03d", dir,1);
			write_mltm_model(corpus, model, ss, var, filename, 0);
			fprintf(lhood_fptr, "%d %e %lf %5ld %lf %lf %lf\n",
					iteration, lkh, rho, (int)t2-t1 + t0, ccr,mcstderr, avg_mcstderr);
			fflush(lhood_fptr);
		}

		time(&t2);
		if (iteration % 500 == 0){
			printf("***** VB ITERATION %d *****, lkh = %lf, "
					"prevlkh = %e, conv=%e, %lf\n", iteration, lkh, prevlkh, conv, mcstderr);
		}
		iteration += 1;
		if ((ready_to_break == 1) && (iteration > 7000))
			break;
		if (iteration > model->T)
			break;

	}while(1);

	fclose(lhood_fptr);

	//last run
	lkh = 0;
	# pragma omp parallel shared (corpus, model, ss, var_array) \
	private (d, thread_id) \
	reduction ( + : lkh)
	{
		thread_id = omp_get_thread_num();
		# pragma omp for schedule(dynamic)
		for (d = 0; d < corpus->ndocs; d++){

			doclkh = testlda_doc_estep(&(corpus->docs[d]), model,
					ss, &(var_array->var[thread_id]), d, thread_id, 1000);

			lkh += doclkh;

		}
	}

	for (j = 0; j < model->m; j++){
		for (n = 0; n < model->n; n++){
			lkh += lgamma(var->mu[j][n]);
		}
		lkh -= lgamma(var->summu[j]);
	}

	sprintf(filename, "%s/final", dir);

	write_mltm_model(corpus, model, ss, var, filename, 0);



}



double doc_estep(document* doc, mltm_model* model, mltm_ss* ss, mltm_var* var,
		int d, int thread_id, int lbld, int iteration){

	int s, i, j, c, n, prev_j, chk, s2, jj=0, ii;
	int cj_ind, nj_ind;//, counter;
	int Ls, Sd, ntopics, nclasses;
	double normsum, u;//, log_Sd;
	double temp, maxval;//, tempgrad, maxval;
	double cdf, avg_rho;
	sentence* sent;
	int s_star;
	double temp_s_star;

	Sd = doc->length;
	ntopics = model->m;
	nclasses = model->c;


	//initialize phi
	int thread_nj = thread_id*model->n*ntopics;
	int thread_j = thread_id*ntopics;
	int iter, tt=-1;
	double rho;
	int rep = 0;
	int thread_cj = thread_id*nclasses*ntopics;

	doc->iteration += 1.0;
	rho = model->s*pow(1.0/(doc->iteration+model->TAU), model->KAPPA);
	avg_rho = rho;

	if (doc->iteration > model->BurnIn)
		tt = (int)(doc->iteration-model->BurnIn)%model->b;

	//recompute sentence labels
	if (lbld == 1){
		for (s = 0; s < doc->length; s++){
			sent = &(doc->sents[s]);
			for (c = 0; c < nclasses; c++){
				ii = sent->xsamples[c];
				jj = sent->samples[ii];
				sent->py[c] = model->mu[c][jj];

				if (s > 0){
					if (sent->py[c] > doc->maxy[c]){
						doc->maxy[c] = sent->py[c];
						doc->argmaxy[c] = s;
					}
				}else{
					doc->maxy[c] = sent->py[c];
					doc->argmaxy[c] = s;
				}
			}
		}
	}

	for (j = 0; j < ntopics; j++){
		var->logm[j] = log(model->alpha + doc->m[j]);
		var->sumphi[j] = 0.0;
	}

	for (iter = 0; iter < doc->rep+rep; iter++){ //for docs which are selected >1 in a batch (to avoid race in openmp)
	

		if (iter >= rep+1){
			doc->iteration += 1.0;
			rho = model->s*pow(1.0/(doc->iteration+model->TAU), model->KAPPA);
			avg_rho += rho;
		}

		//varlkh = 0.0;
		//sj_ind = 0;
		for (s = 0; s < Sd; s++){
			//sj_ind = s*ntopics;
			//sc_ind = s*nclasses;
			sent = &(doc->sents[s]);
			Ls = sent->length;
			//Lsinv = 1.0/((double)sent->length);

			for (i = 0; i < Ls; i++){
				n = sent->words[i];
				nj_ind = n*ntopics;
				//cumind = sent->cumlen + i;

				normsum = 0.0;
				maxval = -1e100;
				//cumind_topic = sent->cumlen_topic + i*ntopics; //add j later

				prev_j = sent->samples[i];
				doc->m[prev_j] -= 1.0;
				var->logm[prev_j] = log(model->alpha + doc->m[prev_j]);

				for (j = 0; j < ntopics; j++){

					var->Mphi[j] = var->logm[j] +  model->Elogbeta[nj_ind + j];

					if (lbld == 1){

						for (c = 0; c < nclasses; c++){

							if (sent->xsamples[c] != i){ //this is not the anchor word for class c
								s_star = doc->argmaxy[c];
								temp_s_star = doc->maxy[c];
							}
							else{
								temp = model->mu[c][j];
								if (s == doc->argmaxy[c]){
									if (temp > doc->maxy[c]){ //this beats the prvious max
										s_star = s;
										temp_s_star = temp;
									}else{ // search
										temp_s_star = temp;//-1e100;
										s_star = s;
										for (s2 = 0; s2 < doc->length; s2++){
											if ((s2 != s) && (doc->sents[s2].py[c] > temp_s_star)){
												temp_s_star = doc->sents[s2].py[c];
												s_star = s2;
											}
											/*if (s2 == s){ //current sent
												if (temp > temp_s_star){
													temp_s_star = temp;
													s_star = s2;
												}
											}else{ // which one is?
												if (doc->sents[s2].py[c] > temp_s_star){
													temp_s_star = doc->sents[s2].py[c];
													s_star = s2;
												}
											}*/
										}
									}
								}else{// this sent was not the max
									if (temp > doc->maxy[c]){ //beats the current max
										s_star = s;
										temp_s_star = temp;
									}else{ //nochange
										s_star = doc->argmaxy[c];
										temp_s_star = doc->maxy[c];
									}
								}

								var->cj_temp_argmaxy[c][j] = s_star;
								var->cj_temp_maxy[c][j] = temp_s_star;
							}


							if (doc->b[c] == 1){
								var->Mphi[j] += log(temp_s_star);
							}else{
								var->Mphi[j] += log(1-temp_s_star);
							}

						} //end of loop over c

					} // end of computing lbl gradients

					if (var->Mphi[j] > maxval)
						maxval = var->Mphi[j];
				}
				normsum = 0.0;
				for (j = 0; j < ntopics; j++){
					var->Mphi[j] = exp(var->Mphi[j] - maxval);
					normsum += var->Mphi[j];
				}

				u = gsl_ran_flat(&(rng[thread_id]), 0, 1);
				chk = 0;
				cdf = 0.0;
				for (j = 0; j < ntopics; j++){

					//temp_cumind_topic = cumind_topic + j;
					var->Mphi[j] /= normsum;

					//if (doc->iteration > model->BurnIn)
					temp = (1-rho)*sent->phi[i][j] + rho*var->Mphi[j];
					//else
					//	temp = var->Mphi[j];

					//var->phi[temp_cumind_topic] = temp;

					sent->phi[i][j] = temp;
					var->sumphi[j] += temp;
					cdf += temp;


					if (chk == 0){
						if (u <= cdf){
							chk = 1;
							jj = j;
						}
					}

					//update ss
					if (iter >= rep){

						if (model->BATCHSIZE == model->D){
							ss->beta_thread[thread_nj + nj_ind+j] += temp;
							ss->sumbeta_thread[thread_j + j] += temp;
						}else{
							if (lbld == 1){
								ss->beta_thread[thread_nj + nj_ind+j] += temp*model->labeled_ratio;
								ss->sumbeta_thread[thread_j + j] += temp*model->labeled_ratio;
							}else{
								ss->beta_thread[thread_nj + nj_ind+j] += temp*(1-model->labeled_ratio);
								ss->sumbeta_thread[thread_j + j] += temp*(1-model->labeled_ratio);
							}
						}
					}

				}


				sent->samples[i] = jj;
				doc->m[jj] += 1.0;
				var->logm[jj] = log(model->alpha + doc->m[jj]);
				j = jj;

				if (lbld == 1){
					for (c = 0; c < nclasses; c++){
						cj_ind = c*ntopics;

						if (sent->xsamples[c] != i){
							continue;
						}
						/*else{
							temp = model->mu[c][j];
							sent->py[c] = temp;
							if (s == doc->argmaxy[c]){
								if (temp > doc->maxy[c]){ //nochange
									s_star = s;
									temp_s_star = temp;
								}else{ // search
									temp_s_star = temp;//-1e100;
									s_star = s;
									for (s2 = 0; s2 < doc->length; s2++){
										if ((s2 != s) && (doc->sents[s2].py[c] > temp_s_star)){
											temp_s_star = doc->sents[s2].py[c];
											s_star = s2;
										}
									}
								}
							}else{// this sent was not the max
								if (temp > doc->maxy[c]){ //beats the current max
									s_star = s;
									temp_s_star = temp;
								}else{ //nochange
									s_star = doc->argmaxy[c];
									temp_s_star = doc->maxy[c];
								}
							}
						}

						*/
						doc->maxy[c] = var->cj_temp_maxy[c][j];
						doc->argmaxy[c] = var->cj_temp_argmaxy[c][j];

						//not using these anymore.
						//doc->logpb[c] = log(temp_s_star);
						//doc->logp1mb[c] = log(1-temp_s_star);

					}
				}
				//MC std er estimation
				if ((iter == doc->rep+rep-1) && (doc->iteration > model->BurnIn)){
					sent->MCz[i] = ((doc->iteration-1.0-model->BurnIn)*sent->MCz[i] +
							(double)j)/(doc->iteration-model->BurnIn);

					if (tt > 0)
						sent->YbarMCz[i] = ((tt-1.0)*sent->YbarMCz[i] +
								(double)j)/tt;
					else{
						sent->YbarMCz[i] = ((model->b-1.0)*sent->YbarMCz[i] +
								(double)j)/model->b;
						doc->zybar2 += pow(sent->YbarMCz[i], 2.0);
						doc->zhat2 += pow(sent->MCz[i], 2.0);
						doc->znum += 1.0;

					}
				}

			}
			//update the anchor words
			if (lbld == 1){
				for (c = 0; c < nclasses; c++){
					maxval = -1e100;
					for (i = 0; i < Ls; i++){
						//trial set this word to be the anchor
						j = sent->samples[i];
						temp = model->mu[c][j];
						if (s == doc->argmaxy[c]){
							if (temp > doc->maxy[c]){ //nochange
								s_star = s;
								temp_s_star = temp;
							}else{ // search
								temp_s_star = temp;//-1e100;
								s_star = s;
								for (s2 = 0; s2 < doc->length; s2++){
									if ((s2 != s) && (doc->sents[s2].py[c] > temp_s_star)){
										temp_s_star = doc->sents[s2].py[c];
										s_star = s2;
									}
									/*if (s2 == s){ //current sent
										if (temp > temp_s_star){
											temp_s_star = temp;
											s_star = s2;
										}
									}else{ // which one is?
										if (doc->sents[s2].py[c] > temp_s_star){
											temp_s_star = doc->sents[s2].py[c];
											s_star = s2;
										}
									}*/
								}
							}
						}else{// this sent was not the max
							if (temp > doc->maxy[c]){ //beats the current max
								s_star = s;
								temp_s_star = temp;
							}else{ //nochange
								s_star = doc->argmaxy[c];
								temp_s_star = doc->maxy[c];
							}
						}
						var->temp_maxy[i] = temp_s_star;
						var->temp_argmaxy[i] = s_star;

						if (doc->b[c] == 1){
							var->px[i] = log(temp_s_star);
						}else{
							var->px[i] = log(1-temp_s_star);
						}

						if (var->px[i] > maxval)	maxval = var->px[i];
					}
					normsum = 0.0;
					for (i = 0; i < Ls; i++){
						var->px[i] = exp(var->px[i] - maxval);
						normsum += var->px[i];
					}
					cdf = 0.0;
					chk = 0;
					u = gsl_ran_flat(&(rng[thread_id]), 0, 1);

					for (i = 0; i < Ls; i++){
						var->px[i] /= normsum;
						//if (doc->iteration > model->BurnIn)
						sent->x[c][i] = (1-rho)*sent->x[c][i] + rho*var->px[i];
						//else
						//	sent->x[c][i] = var->px[i];

						cdf += sent->x[c][i];
						if (chk == 0){
							//if (j > 0)	var->cdfphi[j] = var->cdfphi[j-1] + temp;
							//else var->cdfphi[j] = temp;
							if (u <= cdf){
								chk = 1;
								ii = i;
							}
						}
					}

					jj = sent->samples[ii];
					sent->xsamples[c] = ii;
					temp = model->mu[c][jj];
					//may need to update sent->py and doc->pb
					sent->py[c] = temp;

					/*if (s == doc->argmaxy[c]){
						if (temp > doc->maxy[c]){ //nochange
							s_star = s;
							temp_s_star = temp;
						}else{ // search
							temp_s_star = temp;//-1e100;
							s_star = s;
							for (s2 = 0; s2 < doc->length; s2++){
								if ((s2 != s) && (doc->sents[s2].py[c] > temp_s_star)){
									temp_s_star = doc->sents[s2].py[c];
									s_star = s2;
								}
							}
						}
					}else{// this sent was not the max
						if (temp > doc->maxy[c]){ //beats the current max
							s_star = s;
							temp_s_star = temp;
						}else{ //nochange
							s_star = doc->argmaxy[c];
							temp_s_star = doc->maxy[c];
						}
					}*/

					doc->maxy[c] = var->temp_maxy[ii];
					doc->argmaxy[c] = var->temp_argmaxy[ii];

					//doc->logpb[c] = log(temp_s_star);
					//doc->logp1mb[c] = log(1-temp_s_star);
				}
			}
			//end of anchor word update

		}//end of sentence


		normsum = 0.0;
		for (j = 0; j < ntopics; j++){
			doc->theta[j] = var->sumphi[j];
			var->sumphi[j] = 0.0;
			normsum += doc->theta[j];
		}
		for (j = 0; j < ntopics; j++){
			doc->theta[j] /= normsum;
			var->sumphi[j] = 0.0;
		}


		// update ss
		if ((lbld == 1) && (iter >= rep)){
			//compute gradients for updating w
			for (c = 0; c < nclasses; c++){
				cj_ind = c*ntopics;
				sent = &(doc->sents[doc->argmaxy[c]]);

				j = sent->samples[sent->xsamples[c]];
				//ss->hessw_thread[thread_cj + cj_ind + j] -= model->mu[c][j]*(1-model->mu[c][j]);
				if (doc->b[c] == 1){
					//ss->gradw_thread[thread_cj + cj_ind + j] += 1-model->mu[c][j];
					ss->gradw_on_thread[thread_cj + cj_ind + j] += 1.0;
				}else{
					//ss->gradw_thread[thread_cj + cj_ind + j] -= model->mu[c][j];
					ss->gradw_off_thread[thread_cj + cj_ind + j] += 1.0;
				}

			}

		}
		if (iter >= rep){
			ss->alpha[thread_id] += ntopics*(gsl_sf_psi(model->alpha*model->m)-gsl_sf_psi(model->alpha));
			ss->alpha[thread_id] -= ntopics*gsl_sf_psi(model->alpha*ntopics+doc->mbar);
			for (j = 0; j < ntopics; j++){
				ss->alpha[thread_id] += gsl_sf_psi(model->alpha + doc->m[j]);
			}
		}

	}
	if (tt == 0){
		doc->zybar2 /= doc->znum;
		doc->zhat2 /= doc->znum;
		doc->zmeanybar2 = ((doc->a-1.0)*doc->zmeanybar2 + doc->zybar2)/doc->a;
		doc->mcse_z = model->b*(doc->zmeanybar2 - doc->zhat2);
		doc->mcse_z = sqrt(doc->mcse_z/(doc->iteration-model->BurnIn));
		doc->a += 1;
	}




	return(avg_rho/doc->rep);

}

mltm_var_array* new_mltm_var(mltm_corpus* corpus, mltm_model* model)
{
	int maxsize;
	int s, c, j, n, num;

	mltm_var_array* var_array = malloc(sizeof(mltm_var_array));

	maxsize = corpus->max_length*model->m;

	var_array->var = (mltm_var *) malloc(sizeof(mltm_var)*(nthreads));
	for (num = 0; num < nthreads; num++){

		var_array->var[num].adam_m_alphahat = 0.0;
		var_array->var[num].adam_v_alphahat = 0.0;
		var_array->var[num].adam_m_nuhat = 0.0;
		var_array->var[num].adam_v_nuhat = 0.0;
		var_array->var[num].adam_m_psi1hat = 0.0;
		var_array->var[num].adam_v_psi1hat = 0.0;
		var_array->var[num].adam_m_psi2hat = 0.0;
		var_array->var[num].adam_v_psi2hat = 0.0;

		var_array->var[num].px = malloc(sizeof(double)*corpus->max_length); //should be max # sent
		var_array->var[num].py_ex = malloc(sizeof(double)*corpus->max_length); //should be max # sent
		var_array->var[num].py_ex2 = malloc(sizeof(double)*corpus->max_length); //should be max # sent

		var_array->var[num].sumgamma = 0.0;
		var_array->var[num].gamma = malloc(sizeof(double)*model->m);
		var_array->var[num].tempphi = malloc(sizeof(double)*model->m);
		var_array->var[num].sumphi = malloc(sizeof(double)*model->m);
		var_array->var[num].logm = malloc(sizeof(double)*model->m);
		//var_array->var[num].Elogtheta = malloc(sizeof(double)*model->m);
		var_array->var[num].psigamma = malloc(sizeof(double)*model->m);
		//var_array->var[num].cdfphi = malloc(sizeof(double)*model->m);
		var_array->var[num].oldphi = malloc(sizeof(double)*model->m);
		var_array->var[num].Mphi = malloc(sizeof(double)*model->m);
		for (j = 0; j < model->m; j++){
			var_array->var[num].Mphi[j] = 0.0;
			var_array->var[num].logm[j] = 0.0;
			var_array->var[num].tempphi[j] = 0.0;
			var_array->var[num].gamma[j] = 0.0;
			var_array->var[num].sumphi[j] = 0.0;
			//var_array->var[num].Elogtheta[j] = 0.0;
			var_array->var[num].psigamma[j] = 0.0;
			//var_array->var[num].cdfphi[j] = 0.0;
			var_array->var[num].oldphi[j] = 0.0;
		}
		/*var_array->var[num].phi = malloc(sizeof(double*)*corpus->max_length);
		for (s = 0; s < corpus->max_length; s++){
			var_array->var[num].phi[s] = malloc(sizeof(double)*model->m);
			for (j = 0; j < model->m; j++){
				var_array->var[num].phi[s][j] = 0.0;
			}
		}*/
		var_array->var[num].phi = malloc(sizeof(double)*maxsize);
		var_array->var[num].phibar = malloc(sizeof(double)*maxsize);
		for (s = 0; s < maxsize; s++){
			var_array->var[num].phi[s] = 0.0;
			var_array->var[num].phibar[s] = 0.0;
		}

		var_array->var[num].mu = malloc(sizeof(double*)*model->m); //[j][n]
		var_array->var[num].mu_adam_v = malloc(sizeof(double*)*model->m); //[j][n]
		var_array->var[num].mu_adam_m = malloc(sizeof(double*)*model->m); //[j][n]
		var_array->var[num].gbar = malloc(sizeof(double*)*model->m);
		var_array->var[num].summu = malloc(sizeof(double)*model->m); //[j]
		var_array->var[num].hbar = malloc(sizeof(double)*model->m);
		var_array->var[num].rho = malloc(sizeof(double)*model->m);
		var_array->var[num].tau = malloc(sizeof(double)*model->m);
		for (j = 0; j < model->m; j++){
			var_array->var[num].rho[j] = 0.0;
			var_array->var[num].tau[j] = 0.0;
			var_array->var[num].summu[j] = 0.0;
			var_array->var[num].hbar[j] = 0.0;
			var_array->var[num].mu[j] = malloc(sizeof(double)*model->n);
			var_array->var[num].mu_adam_m[j] = malloc(sizeof(double)*model->n);
			var_array->var[num].mu_adam_v[j] = malloc(sizeof(double)*model->n);
			var_array->var[num].gbar[j] = malloc(sizeof(double)*model->n);
			for (n = 0; n < model->n; n++){
				var_array->var[num].mu[j][n] = 0.0;
				var_array->var[num].mu_adam_m[j][n] = 0.0;
				var_array->var[num].mu_adam_v[j][n] = 0.0;
				var_array->var[num].gbar[j][n] = 0.0;
			}
		}

		/*var_array->var[num].gradw = malloc(sizeof(double)*model->c*model->m);
		for (c = 0; c < model->m*model->c; c++){
			var_array->var[num].gradw[c] = 0.0;
		}*/
		var_array->var[num].adam_m = malloc(sizeof(double*)*model->c);
		var_array->var[num].adam_v = malloc(sizeof(double*)*model->c);
		for (c = 0; c < model->c; c++){
			var_array->var[num].adam_m[c] = malloc(sizeof(double)*model->m);
			var_array->var[num].adam_v[c] = malloc(sizeof(double)*model->m);
			for (j = 0; j < model->m; j++){
				var_array->var[num].adam_m[c][j] = 0.0;
				var_array->var[num].adam_v[c][j] = 0.0;
			}
		}

		var_array->var[num].grad_temp = malloc(sizeof(double)*model->m);
		var_array->var[num].grad_temp_init = malloc(sizeof(int)*model->m);
		for (j = 0; j < model->m; j++){
			var_array->var[num].grad_temp[j] = 0.0;
			var_array->var[num].grad_temp_init[j] = 0;
		}

		var_array->var[num].temp_argmaxy = malloc(sizeof(int)*corpus->max_length);
		var_array->var[num].temp_maxy = malloc(sizeof(double)*corpus->max_length);
		for (s = 0; s < corpus->max_length; s++){
			var_array->var[num].temp_argmaxy[s] = 0;
			var_array->var[num].temp_maxy[s] = 0.0;
		}

		var_array->var[num].cj_temp_argmaxy = malloc(sizeof(int*)*model->c);
		var_array->var[num].cj_temp_maxy = malloc(sizeof(double*)*model->c);
		for (c = 0; c < model->c; c++){
			var_array->var[num].cj_temp_argmaxy[c] = malloc(sizeof(int)*model->m);
			var_array->var[num].cj_temp_maxy[c] = malloc(sizeof(double)*model->m);
			for (j = 0; j < model->m; j++){
				var_array->var[num].cj_temp_argmaxy[c][j] = 0;
				var_array->var[num].cj_temp_maxy[c][j] = 0.0;
			}
		}
	}
	return(var_array);
}


void test(char* dataset, char* settings_file, char* model_name, char* dir)
{

	FILE* lhood_fptr;
	FILE* fp;
	char string[100];
	char filename[100];
	int ntopics, nclasses, ndocs;
	double doclkh, temp;
	int thread_id;
	//double rho;
	//double doclkh, wrdlkh;
	//double normsum, temp, psi_sigma, alpha_sigma, nu_sigma;
	//int iteration;
	int d, n, j;

	mltm_corpus* corpus;
	mltm_model *model = NULL;
	mltm_ss *ss = NULL;
	//mltm_phiopt *phiopt = NULL;
	mltm_var *var = NULL;
	mltm_var_array *var_array = NULL;

	time_t t1,t2;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &ndocs);
	fscanf(fp, "N %d\n", &d);
	fscanf(fp, "T %d\n", &MAXITER);
	fscanf(fp, "L %d\n", &L0);
	//fscanf(fp, "alpha %lf\n", &L0);
	fclose(fp);

	corpus = read_data(dataset, nclasses, ndocs, 0, filename, ntopics);
	model = new_mltm_model(corpus, settings_file);
	model->L = 1;
	ss = new_mltm_ss(model);
	var_array = new_mltm_var(corpus, model);

	var = &(var_array->var[0]);


	corpus->nterms = model->n;

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file
	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

	load_model(model, corpus, ss, var, model_name);

	for (j = 0; j < model->m; j++){
		var->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			var->summu[j] += var->mu[j][n];
			for (thread_id = 1; thread_id < nthreads; thread_id++){
				var_array->var[thread_id].mu[j][n] = var->mu[j][n];
			}
		}
		for (thread_id = 1; thread_id < nthreads; thread_id++){
			var_array->var[thread_id].summu[j] = var->summu[j];
		}
		temp = gsl_sf_psi(var->summu[j]);
		for (n = 0; n < model->n; n++){

			model->Elogbeta[n*ntopics+j] = (gsl_sf_psi(var->mu[j][n])-temp);

			//model->expElogbeta[j][n] = exp(model->Elogbeta[n*ntopics+j]);
		}

	}

	//iteration = 0;

	time(&t1);


	double wrdlkh = 0.0;
	# pragma omp parallel shared (corpus, model, ss, var_array) \
	private (d, thread_id) \
	reduction ( + : wrdlkh)
	{
		thread_id = omp_get_thread_num();
		# pragma omp for schedule(dynamic)
		for (d = 0; d < corpus->ndocs; d++){

			doclkh = testlda_doc_estep(&(corpus->docs[d]), model,
					ss, &(var_array->var[thread_id]), d, 0, L0);

			//lkh += doclkh;

			wrdlkh += compute_wrdlkh(&(corpus->docs[d]), model, ss, &(var_array->var[thread_id]), d);

		}
	}
	time(&t2);

	fprintf(lhood_fptr, "%d %e %lf %5ld\n", 0, wrdlkh, 0.0, (int)t2-t1);
	fclose(lhood_fptr);

	sprintf(filename, "%s/testfinal", dir);
	write_mltm_model(corpus, model, ss, var, filename, 1);


}

double compute_wrdlkh(document* doc, mltm_model* model, mltm_ss* ss, mltm_var* var, int d){

	int s, i, j, n;
	double temp, lkh;
	sentence * sent;
	lkh = 0.0;
	for (s = 0; s < doc->length; s++){
		sent = &(doc->sents[s]);
		temp = 0.0;
		for (i = 0; i < sent->length; i++){
			n = sent->words[i];
			//nj_ind = n*model->m;
			temp = 0.0;
			for (j = 0; j < model->m; j++){
				temp += doc->theta[j]*(var->mu[j][n]/var->summu[j]);
			}
			lkh += log(temp);
		}
	}
	return(lkh);
}


double testlda_doc_estep(document* doc, mltm_model* model, mltm_ss* ss,
		mltm_var* var, int d, int thread_id, int Lmax){

	int s, i, j, c, n, nj_ind;
	int variter, cumind_topic, temp_cumind_topic;
	double varlkh, prev_varlkh, normsum;
	double temp, maxval;
	double conv;
	int ntopics = model->m;
	int Sd = doc->length;
	int Ls;

	sentence* sent;

	for (j = 0; j < ntopics; j++){
		var->sumphi[j] = 0.0;
	}
	var->sumgamma = 0.0;
	for (s = 0; s < Sd; s++){
		sent = &(doc->sents[s]);
		Ls = sent->length;
		for (i = 0; i < Ls; i++){
			n = sent->words[i];
			nj_ind = n*ntopics;
			//cumind = sent->cumlen + i;
			cumind_topic = sent->cumlen_topic + i*ntopics;
			maxval = -1e50;
			for (j = 0; j < ntopics; j++){
				var->phi[cumind_topic + j] = model->Elogbeta[nj_ind + j];
				if (var->phi[cumind_topic + j] > maxval)
					maxval = var->phi[cumind_topic + j];
			}
			normsum = 0.0;
			for (j = 0; j < ntopics; j++){
				var->phi[cumind_topic + j] = exp(var->phi[cumind_topic + j] - maxval);
				normsum += var->phi[cumind_topic + j];
			}
			for (j = 0; j < ntopics; j++){
				var->phi[cumind_topic + j] /= normsum;
				//var->phi[cumind_topic + j] = 1.0/((double)model->m);
				var->sumphi[j] += var->phi[cumind_topic + j];
				var->sumgamma += var->phi[cumind_topic + j];

			}
		}
	}
	var->sumgamma += model->alpha*ntopics;
	//temp = gsl_sf_psi(var->sumgamma);
	for (j = 0; j < ntopics; j++){
		var->gamma[j] = model->alpha + var->sumphi[j];
		var->psigamma[j] = gsl_sf_psi(var->gamma[j]);
		//var->Elogtheta[j] = var->psigamma[j] - temp;
	}

	//update phi
	variter = 0;
	prev_varlkh = -1e100;

	do{

		varlkh = 0.0;

		for (s = 0; s < Sd; s++){
			sent = &(doc->sents[s]);
			Ls = sent->length;
			//Sdinv = 1.0/((double)sent->length);
			for (i = 0; i < Ls; i++){
				n = sent->words[i];
				nj_ind = n*ntopics;
				//cumind = sent->cumlen + i;
				normsum = 0.0;
				maxval = -1e100;
				cumind_topic = sent->cumlen_topic + i*ntopics; //add j later
				for (j = 0; j < ntopics; j++){
					var->oldphi[j] = var->phi[cumind_topic + j];

					var->tempphi[j] = var->psigamma[j]+model->Elogbeta[nj_ind + j];

					if (var->tempphi[j] > maxval)
						maxval = var->tempphi[j];
				}
				normsum = 0.0;
				for (j = 0; j < ntopics; j++){
					var->tempphi[j] = exp(var->tempphi[j] - maxval);
					normsum += var->tempphi[j];
				}
				for (j = 0; j < ntopics; j++){

					temp_cumind_topic = cumind_topic + j;
					var->phi[temp_cumind_topic] = var->tempphi[j]/normsum;

					temp = var->phi[temp_cumind_topic] - var->oldphi[j];
					var->sumphi[j] += temp;
					var->gamma[j] += temp;
					var->sumgamma += temp;
					var->psigamma[j] = gsl_sf_psi(var->gamma[j]);

					//update lkh
					if (var->phi[temp_cumind_topic] > 0) //the gamma part cancels
						varlkh += var->phi[temp_cumind_topic]*(model->Elogbeta[nj_ind + j]-log(var->phi[temp_cumind_topic]));
				}

			}
		}
		//update varlkh


		varlkh -= lgamma(var->sumgamma); //the other terms now ARE cancelled
		//varlkh += lgamma(model->alpha*model->m)-lgamma(model->alpha)*model->m;
		temp = gsl_sf_psi(var->sumgamma);
		for (j = 0; j < ntopics; j++){
			varlkh += lgamma(var->gamma[j]);// +
					//(model->alpha-var->gamma[j]+var->sumphi[j])*(var->psigamma[j]-temp);
		}

		conv = fabs(varlkh - prev_varlkh)/fabs(prev_varlkh);

		if ((conv < 1e-4) || (variter > 100))
			break;

		prev_varlkh = varlkh;
		variter += 1;


	}while(1);


	double maxi;
	//double log_Sd = log((double)doc->length);
	//double Pinv = 1.0/model->p;
	//int cj_ind;
	/*for (c = 0; c < model->c; c++){
		//cj_ind = c*model->m;
		doc->pb[c] = -1e100;
		for (s = 0; s < doc->length; s++){
			sent = &(doc->sents[s]);
			maxi = -1e50;
			for (i = 0; i < sent->length; i++){
				//cumind = sent->cumlen + i;
				cumind_topic = sent->cumlen_topic + i*model->m;
				temp = 0.0;
				for (j = 0; j < model->m; j++){
					temp += var->phi[cumind_topic + j]*model->mu[c][j];
				}
				if (temp > maxi)	maxi = temp;
			}
			sent->maxi[c] = maxi;
			if (sent->maxi[c] > doc->pb[c])
				doc->pb[c] = sent->maxi[c];
		}
	}*/

	double pb, u, cdf, Linv;
	double Ex, Ex2, mcer;
	int mcerChk, l;

	for (c = 0; c < model->c; c++){
		//cj_ind = c*model->m;
		doc->pb[c] = 0.0;
		Ex = 0.0;
		Ex2 = 0.0;
		mcerChk = 0;
		for (l = 0; l < Lmax; l++){
			pb = -1e100;
			for (s = 0; s < Sd; s++){
				sent = &(doc->sents[s]);
				Ls = sent->length;
				Linv = 1.0/((double)Ls);
				/*sent->py[c] = 0.0;
				for (i = 0; i < sent->length; i++){
					cumind_topic = sent->cumlen_topic + i*model->m;
					for (j = 0; j < model->m; j++){
						sent->py[c] += Linv*var->phi[cumind_topic + j]*model->mu[c][j];
					}
				}*/

				//maxi = -1e100;
				//choose an anchor word
				u = gsl_ran_flat(&(rng[thread_id]), 0, 1);
				cdf = 0.0;
				for (i = 0; i < Ls; i++){
					cdf += Linv;
					if (u < cdf)	break;
				}
				// choose a topic
				cumind_topic = sent->cumlen_topic + i*ntopics;
				u = gsl_ran_flat(&(rng[thread_id]), 0, 1);
				cdf = 0.0;
				for (j = 0; j < ntopics; j++){
					cdf += var->phi[cumind_topic + j];
					if (u < cdf)	break;
				}
				maxi = model->mu[c][j];
				if (l == 0){
					sent->py[c] = maxi;
					var->py_ex[s] = maxi;
					var->py_ex2[s] = maxi*maxi;
				}
				else{
					sent->py[c] = (sent->py[c]*l + maxi)/(l+1.0);
					var->py_ex[s] = (var->py_ex[s]*l + maxi)/(l+1.0);
					var->py_ex2[s] = (var->py_ex2[s]*l + maxi*maxi)/(l+1.0);
				}
				if (l > 1000){
					mcer = sqrt(var->py_ex2[s]-var->py_ex[s]*var->py_ex[s])/sqrt(l);
					if (mcer > 0.01)
						mcerChk = 1;
				}

				if (maxi > pb)	pb = maxi;
			}

			//if (l == 0)
			//	doc->pb[c] = pb;
			//else
			//	doc->pb[c] = (l*doc->pb[c] + pb)/(l+1.0);
			Ex = (l*Ex + pb)/(l+1.0);
			Ex2 = (l*Ex + pb*pb)/(l+1.0);
			if (l > 1000){
				mcer = sqrt(Ex2 - Ex*Ex)/sqrt(l);
				if (mcer > 0.01)
					mcerChk = 1;
			}
			if ((l > 1000) && (mcerChk == 0)){
				break;
			}
		}
		doc->pb[c] = Ex;
		//printf("%d %d\n", c, l);
		//doc->pb[c] /= 100.0;
		//doc->pb[c] = 1 - pow(doc->pb[c]/((double)doc->length), 1.0/model->p);
	}

	/*double pb, u, cdf, Linv;
	double Ex, Ex2, mcer;
	int mcerChk, l;

	for (c = 0; c < model->c; c++){
		//cj_ind = c*model->m;
		doc->pb[c] = 0.0;
		Ex = 0.0;
		Ex2 = 0.0;
		mcerChk = 0;
		for (l = 0; l < 10000; l++){
			pb = -1e100;
			for (s = 0; s < doc->length; s++){
				sent = &(doc->sents[s]);
				Linv = 1.0/((double)sent->length);
				//maxi = -1e100;
				//choose an anchor word
				u = gsl_ran_flat(&(rng[thread_id]), 0, 1);
				cdf = 0.0;
				for (i = 0; i < sent->length; i++){
					cdf += Linv;
					if (u < cdf)	break;
				}
				// choose a topic
				cumind_topic = sent->cumlen_topic + i*model->m;
				u = gsl_ran_flat(&(rng[thread_id]), 0, 1);
				cdf = 0.0;
				for (j = 0; j < model->m; j++){
					cdf += var->phi[cumind_topic + j];
					if (u < cdf)	break;
					//printf("(%lf, %lf) ",var->phi[cumind][j], temp);
				}
				maxi = model->mu[c][j];
				if (l == 0){
					sent->py[c] = maxi;
					var->py_ex[s] = maxi;
					var->py_ex2[s] = maxi*maxi;
				}
				else{
					sent->py[c] = (sent->py[c]*l + maxi)/(l+1.0);
					var->py_ex[s] = (var->py_ex[s]*l + maxi)/(l+1.0);
					var->py_ex2[s] = (var->py_ex2[s]*l + maxi*maxi)/(l+1.0);
				}
				if (l > 1000){
					mcer = sqrt(var->py_ex2[s]-var->py_ex[s]*var->py_ex[s])/sqrt(l);
					if (mcer > 0.01)
						mcerChk = 1;
				}

				if (maxi > pb)	pb = maxi;
			}

			//if (l == 0)
			//	doc->pb[c] = pb;
			//else
			//	doc->pb[c] = (l*doc->pb[c] + pb)/(l+1.0);
			Ex = (l*Ex + pb)/(l+1.0);
			Ex2 = (l*Ex + pb*pb)/(l+1.0);
			if (l > 1000){
				mcer = sqrt(Ex2 - Ex*Ex)/sqrt(l);
				if (mcer > 0.01)
					mcerChk = 1;
			}
			if ((l > 1000) && (mcerChk == 0)){
				break;
			}
		}
		doc->pb[c] = Ex;
		//printf("%d %d\n", c, l);
		//doc->pb[c] /= 100.0;
		//doc->pb[c] = 1 - pow(doc->pb[c]/((double)doc->length), 1.0/model->p);
	}*/

	//update theta
	for (j = 0; j < ntopics; j++){
		doc->theta[j] = var->gamma[j]/var->sumgamma;//ss->theta[j]/ss->sumtheta;
		/*if (doc->theta[j] == 0)
			doc->theta[j] = EPS;
		doc->logtheta[j] = log(doc->theta[j]);*/
		//printf("%lf\n", doc->theta[j]);
		//ss->theta[j] = 0.0;
	}
	//ss->sumtheta = 0.0;

	return(varlkh);

}

void test_initialize(mltm_corpus* corpus, mltm_model* model, mltm_ss* ss, char* model_name){

/*	int i, j, s, d, n, c;
	double y;
	char filename[400];
	FILE* fileptr;

	sprintf(filename, "%s.beta", model_name);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			fscanf(fileptr, " %lf", &y);
			model->MCbeta[j][n] = y;
			mcmc->logt[j][n] = log(model->MCbeta[j][n]);
		}
	}
	fclose(fileptr);

    //load mu
	sprintf(filename, "%s.mu", model_name);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (j = 0; j < model->m; j++){
		for (c = 0; c < model->c; c++){
			fscanf(fileptr, " %lf", &y);
			model->MCmu[j][c] = y;
			mcmc->logmu[j][c] = log(y);
			mcmc->log1mmu[j][c] = log(1.0-y);
		}
	}
	fclose(fileptr);

	//load psi
	sprintf(filename, "%s.psi", model_name);
	fileptr = fopen(filename, "r");
	for (c = 0; c < model->c; c++){
		fscanf(fileptr, "%lf %lf", &model->psi[c][0], &model->psi[c][1]);
		model->MCpsi[c][0] = model->psi[c][0];
		model->MCpsi[c][1] = model->psi[c][1];
	}
	fclose(fileptr);

	//load alpha
	sprintf(filename, "%s.alpha", model_name);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%lf", &model->alpha);
	fclose(fileptr);

	//load nu
	sprintf(filename, "%s.nu", model_name);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%lf", &model->nu);
	fclose(fileptr);

	//init others

	double* theta = malloc(sizeof(double)*model->m);
	double* p = malloc(sizeof(double)*model->m);
	double* alpha = malloc(sizeof(double)*model->m);
	int* z = malloc(sizeof(int)*model->m);
	int check, maxY;
	double u, normsum;

	for (d = 0; d < corpus->ndocs; d++){
		//init sample theta
		for (j = 0; j < model->m; j++){
			alpha[j] = model->alpha;
		}
		gsl_ran_dirichlet (r, model->m, alpha, theta);
		for (j = 0; j < model->m; j++){
			model->MCtheta[j][d] = theta[j];
		}
		//sample z
		for (s = 0; s < corpus->docs[d].length; s++){
			for (i = 0; i < corpus->docs[d].sents[s].length; i++){
				n = corpus->docs[d].sents[s].words[i];

				for (j = 0; j < model->m; j++){
					//p[j] = 1000.0*model->MCtheta[j][d]*model->MCbeta[j][n];
					p[j] = mcmc->logt[j][n];
					if (j > 0)
						normsum = log_sum(normsum, p[j]);
					else
						normsum = p[j];
				}
				for (j = 0; j < model->m; j++){
					p[j] = exp(p[j] - normsum);
				}

				gsl_ran_multinomial (r, model->m, 1, p, (unsigned int *)z); //p will be normalized in this function
				check = 0;
				for (j = 0; j < model->m; j++){
					if (z[j] == 1){
						corpus->docs[d].sents[s].z[i] = j;
						corpus->docs[d].sents[s].MCz[i] = (double)j;
						mcmc->m[j][d] += 1.0;
						//mcmc->t[j][n] += 1.0;
						//mcmc->tbar[j] += 1.0;
						check = 1;
						break;
					}
				}
				if (check == 0)	printf("logical error. j = %d, d = %d, s = %d, i = %d\n", j, d, s, i);

			}
			//sample x
			for (c = 0; c < model->c; c++){
				check = 0;
				u = gsl_ran_flat (r, 0, 1);
				for (i = 0; i < corpus->docs[d].sents[s].length; i++){
					if (u <= (i+1.0)/((double)corpus->docs[d].sents[s].length)){
						check = 1;
						break;
					}
				}
				if (check == 0)	printf("logical error. i = %d, d = %d, s = %d, Ls = %d\n",
						i, d, s, corpus->docs[d].sents[s].length);
				corpus->docs[d].sents[s].x[c] = i;
				corpus->docs[d].sents[s].MCx[c] = (double)i;
			}

		}

		//sample Ysc
		for (c = 0; c < model->c; c++){

			if (corpus->docs[d].b[c] == 0){
				for (s = 0; s < corpus->docs[d].length; s++){
					corpus->docs[d].sents[s].y[c] = 0;
				}
			}
			else{

				maxY = 0;
				for (s = 0; s < corpus->docs[d].length; s++){
					j = corpus->docs[d].sents[s].z[corpus->docs[d].sents[s].x[c]];
					u = gsl_ran_flat (r, 0, 1);
					if (u < model->MCmu[j][c]){
						corpus->docs[d].sents[s].y[c] = 1;
						maxY = 1;
					}
					else
						corpus->docs[d].sents[s].y[c] = 0;

				}
				//break if maxYsc = bsc
				corpus->docs[d].b[c] = maxY;
				corpus->docs[d].MCb[c] = (double) maxY;

			}

		}

		//update theta
		normsum = 0.0;
		for (j = 0; j < model->m; j++){
			model->MCtheta[j][d] = model->alpha + mcmc->m[j][d];
			mcmc->logm[j][d] = log(model->alpha + mcmc->m[j][d]);
			normsum += model->MCtheta[j][d];
		}
		for (j = 0; j < model->m; j++){
			model->MCtheta[j][d] /= normsum;
		}

	}

	free(theta);
	free(p);
	free(alpha);
	free(z);
*/
}

mltm_model* new_mltm_model(mltm_corpus* corpus, char* settings_file)
{
	int n, c, ntopics, nclasses;
	FILE* fp;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fclose(fp);

	mltm_model* model = malloc(sizeof(mltm_model));
	model->c = nclasses;
	model->m = ntopics;
	model->D = 0;
	model->n = 0;
	model->T = 0;
	model->L = L0;
	model->b = 100;

	//model->adam_eps = 1e-10;

	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &model->D);
	fscanf(fp, "N %d\n", &model->n);
	fscanf(fp, "T %d\n", &model->T);
	fscanf(fp, "L %d\n", &model->L);
	fscanf(fp, "alpha %lf\n", &model->alpha);
	fscanf(fp, "nu %lf\n", &model->nu);
	fscanf(fp, "kappa %lf\n", &model->KAPPA);
	fscanf(fp, "tau %lf\n", &model->TAU);
	fscanf(fp, "batchsize %d\n", &model->BATCHSIZE);
	fscanf(fp, "lag %d\n", &model->lag);
	fscanf(fp, "save_lag %d\n", &model->save_lag);
	fscanf(fp, "psi %lf %lf\n", &model->psi1, &model->psi2);
	fscanf(fp, "rho0 %lf\n", &model->rho0);
	fscanf(fp, "burnin %d\n", &model->BurnIn);
	fscanf(fp, "s %lf\n", &model->s);
	fclose(fp);

	model->alphahat = log(model->alpha);
	model->nuhat = log(model->nu);
	/*model->expElogbeta = malloc(sizeof(double*)*model->m);
	for (j = 0; j < model->m; j++){
		model->expElogbeta[j] = malloc(sizeof(double)*model->n);
		//model->Elogbeta[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			model->expElogbeta[j][n] = 0.0;
			//model->Elogbeta[j][n] = 0.0;
		}
	}*/
	model->Elogbeta = malloc(sizeof(double)*model->m*model->n);
	for (n = 0; n < model->n*model->m; n++){
		model->Elogbeta[n] = 0.0;
	}
	int j;
	/*model->MCbeta = malloc(sizeof(double*)*model->m);
	for (j = 0; j < model->m; j++){
		model->MCbeta[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			model->MCbeta[j][n] = 0.0;
		}
	}*/


	model->mu = malloc(sizeof(double*)*model->c);
	//model->MCw = malloc(sizeof(double*)*model->c);
	//model->muhat = malloc(sizeof(double)*model->c);
	for (c = 0; c < model->c; c++){
		model->mu[c] = malloc(sizeof(double)*model->m);
		//model->MCw[c] = malloc(sizeof(double)*model->m);
		//model->muhat[c] = malloc(sizeof(double)*model->m);
		for (j = 0; j < model->m; j++){
			model->mu[c][j] = 0.0;
			//model->MCw[c][j] = 0.0;
			//model->muhat[c][j] = 0.0;
		}
	}

	return(model);
}

mltm_ss * new_mltm_ss(mltm_model* model){

	int n, j;

	mltm_ss* ss = malloc(sizeof(mltm_ss));

	ss->alpha = malloc(sizeof(double)*nthreads);
	for (j = 0; j < nthreads; j++){
		ss->alpha[j] = 0.0;
	}
	ss->nu = 0.0;

	ss->beta = malloc(sizeof(double)*model->m*model->n);
	for (n = 0; n < model->n*model->m; n++){
		ss->beta[n] = 0.0;
	}
	ss->sumbeta = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		//ss->beta[j] = malloc(sizeof(double)*model->n);
		ss->sumbeta[j] = 0.0;
		/*for (n = 0; n < model->n; n++){
			ss->beta[j][n] = 0.0;
		}*/
	}
	ss->gradw_on = malloc(sizeof(double)*model->c*model->m);
	ss->gradw_off = malloc(sizeof(double)*model->c*model->m);
	for (j = 0; j < model->m*model->c; j++){
		ss->gradw_on[j] = 0.0;
		ss->gradw_off[j] = 0.0;
	}

	//threads
	ss->beta_thread = malloc(sizeof(double)*model->m*model->n*nthreads);
	for (n = 0; n < model->n*model->m*nthreads; n++){
		ss->beta_thread[n] = 0.0;
	}
	ss->sumbeta_thread = malloc(sizeof(double)*model->m*nthreads);
	for (j = 0; j < model->m*nthreads; j++){
		//ss->beta[j] = malloc(sizeof(double)*model->n);
		ss->sumbeta_thread[j] = 0.0;
		/*for (n = 0; n < model->n; n++){
			ss->beta[j][n] = 0.0;
		}*/
	}
	//ss->gradw_thread = malloc(sizeof(double)*model->c*model->m*nthreads);
	ss->gradw_on_thread = malloc(sizeof(double)*model->c*model->m*nthreads);
	ss->gradw_off_thread = malloc(sizeof(double)*model->c*model->m*nthreads);
	//ss->hessw_thread = malloc(sizeof(double)*model->c*model->m*nthreads);
	for (j = 0; j < model->m*model->c*nthreads; j++){
		//ss->gradw_thread[j] = 0.0;
		ss->gradw_on_thread[j] = 0.0;
		ss->gradw_off_thread[j] = 0.0;
		//ss->hessw_thread[j] = 0.0;
	}

	return(ss);
}




mltm_corpus* read_data(const char* data_filename, int nclasses, int ndocs,
		int lblchck, const char* lbl_filename, int ntopics)
{
	FILE *fileptr;
	int Ls, word, nd, nw, lbl;
	int Sd, s, i, c, j, total_length, total_length_topic;
	mltm_corpus* corpus;

	printf("reading data from %s\n", data_filename);
	corpus = malloc(sizeof(mltm_corpus));
	//corpus->docs = 0;
	corpus->docs = (document *) malloc(sizeof(document)*(ndocs));
	corpus->nterms = 0.0;
	corpus->ndocs = 0.0;
	corpus->max_length = 0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;

	while ((fscanf(fileptr, "<%10d> ", &Sd) != EOF)){ // number of sentences
		corpus->docs[nd].iteration = 0.0;
		corpus->docs[nd].sents = (sentence *) malloc(sizeof(sentence)*Sd);
		corpus->docs[nd].length = Sd;
		corpus->docs[nd].b = malloc(sizeof(int)*nclasses);
		corpus->docs[nd].argmaxy = malloc(sizeof(int)*nclasses);
		corpus->docs[nd].maxy = malloc(sizeof(double)*nclasses);
		corpus->docs[nd].pb = malloc(sizeof(double)*nclasses);
		corpus->docs[nd].logpb = malloc(sizeof(double)*nclasses);
		corpus->docs[nd].logp1mb = malloc(sizeof(double)*nclasses);
		for (c = 0; c < nclasses; c++){
			corpus->docs[nd].b[c] = 0;
			corpus->docs[nd].argmaxy[c] = 0;
			corpus->docs[nd].maxy[c] = 0.0;
			corpus->docs[nd].pb[c] = 0.0;
			corpus->docs[nd].logpb[c] = 0.0;
			corpus->docs[nd].logp1mb[c] = 0.0;
		}
		corpus->docs[nd].theta = malloc(sizeof(double)*ntopics);
		corpus->docs[nd].logtheta = malloc(sizeof(double)*ntopics);
		corpus->docs[nd].m = malloc(sizeof(double)*ntopics);
		corpus->docs[nd].mbar = 0.0;
		for (j = 0; j < ntopics; j++){
			corpus->docs[nd].theta[j] = 0.0;
			corpus->docs[nd].logtheta[j] = 0.0;
			corpus->docs[nd].m[j] = 0.0;
		}
		corpus->docs[nd].a = 1; //for batch mean
		corpus->docs[nd].zybar2 = 0.0;
		corpus->docs[nd].zhat2 = 0.0;
		corpus->docs[nd].zmeanybar2 = 0.0;
		corpus->docs[nd].znum = 0;
		corpus->docs[nd].mcse_z = 0.0;

		total_length = 0;
		total_length_topic = 0;
		for (s = 0; s < Sd; s++){
			fscanf(fileptr, "<%10d> ", &Ls);

			corpus->docs[nd].sents[s].x = malloc(sizeof(double*)*nclasses);
			corpus->docs[nd].sents[s].xsamples = malloc(sizeof(int)*nclasses);
			corpus->docs[nd].sents[s].py = malloc(sizeof(double)*nclasses);
			for (c = 0; c < nclasses; c++){
				corpus->docs[nd].sents[s].x[c] = malloc(sizeof(double)*Ls);
				for (i = 0; i < Ls; i++){
					corpus->docs[nd].sents[s].x[c][i] = 0.0;
				}
				corpus->docs[nd].sents[s].xsamples[c] = 0;
				corpus->docs[nd].sents[s].py[c] = 0.0;
			}

			corpus->docs[nd].sents[s].cumlen = total_length;
			corpus->docs[nd].sents[s].cumlen_topic = total_length_topic;
			total_length += Ls;
			total_length_topic += Ls*ntopics;

			corpus->docs[nd].sents[s].MCz = malloc(sizeof(double)*Ls);
			corpus->docs[nd].sents[s].YbarMCz = malloc(sizeof(double)*Ls);

			corpus->docs[nd].sents[s].words = malloc(sizeof(int)*Ls);
			corpus->docs[nd].sents[s].samples = malloc(sizeof(int)*Ls);
			corpus->docs[nd].sents[s].length = Ls;
			corpus->docs[nd].sents[s].phi = malloc(sizeof(double*)*Ls);

			for (i = 0; i < Ls; i++){
				//fscanf(fileptr, "%10d:%10d ", &word, &count);
				fscanf(fileptr, "%10d ", &word);
				corpus->docs[nd].sents[s].words[i] = word;
				corpus->docs[nd].sents[s].samples[i] = 0;
				corpus->docs[nd].sents[s].YbarMCz[i] = 0.0; //for batchmeans
				corpus->docs[nd].sents[s].MCz[i] = 0.0; //for batchmeans
				corpus->docs[nd].sents[s].phi[i] = malloc(sizeof(double)*ntopics);
				for (j = 0; j < ntopics; j++){
					corpus->docs[nd].sents[s].phi[i][j] = 0.0;
				}
				if (word >= nw) { nw = word + 1; }
			}
		}
		nd++;
		if (total_length > corpus->max_length)
			corpus->max_length = total_length;
	}
	fclose(fileptr);
	corpus->ndocs = nd;
	corpus->nterms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);

	if (lblchck == 1){
		printf("reading data from %s\n", lbl_filename);
		fileptr = fopen(lbl_filename, "r");
		for (nd = 0; nd < corpus->ndocs; nd++){
			for (c = 0; c < nclasses; c++){
				fscanf(fileptr, "%d", &lbl);
				corpus->docs[nd].b[c] = lbl;
			}
		}
		fclose(fileptr);
	}
	return(corpus);
}



void write_mltm_model(mltm_corpus * corpus, mltm_model * model, mltm_ss* ss, mltm_var* var,
		char * root, int chktest)
{
	char filename[200];
	FILE* fileptr;
	FILE* fptheta;
	FILE* fpy;
	FILE* fpb;
	int n, j, d, c, s;
	document* doc;
	sentence* sent;

	sprintf(filename, "%s.psi", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "%lf %lf\n", model->psi1, model->psi2);
	fprintf(fileptr, "%lf %lf\n", var->adam_m_psi1hat, var->adam_v_psi1hat);
	fprintf(fileptr, "%lf %lf", var->adam_m_psi2hat, var->adam_v_psi2hat);
	fclose(fileptr);

	sprintf(filename, "%s.alpha", root);
	//printf("loading %s\n", filename);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "%lf %lf\n", model->alphahat, model->alpha);
	fprintf(fileptr, "%Lf %Lf", var->adam_m_alphahat, var->adam_v_alphahat);
	fclose(fileptr);

	sprintf(filename, "%s.nu", root);
	//printf("loading %s\n", filename);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "%lf %lf\n", model->nuhat, model->nu);
	fprintf(fileptr, "%Lf %Lf", var->adam_m_nuhat, var->adam_v_nuhat);
	fclose(fileptr);


	//beta
	sprintf(filename, "%s.beta", root);
	fileptr = fopen(filename, "w");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			//fprintf(fileptr, "%.10lf ",model->Elogbeta[j][n]);
			fprintf(fileptr, "%.10lf ",var->mu[j][n]);
			//fprintf(fileptr, "%.10lf ",model->MCbeta[j][n]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	//w
	sprintf(filename, "%s.w", root);
	fileptr = fopen(filename, "w");
	for (c = 0; c < model->c; c++){
		//cj_ind = c*model->m;
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%5.10lf ", model->mu[c][j]);
			//fprintf(fileptr, "%5.10lf ", model->MCw[c][j]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);


	//if (chktest == 1){

		//Y
		sprintf(filename, "%s.theta", root);
		fptheta = fopen(filename, "w");
		sprintf(filename, "%s.y", root);
		fpy = fopen(filename, "w");
		sprintf(filename, "%s.b", root);
		fpb = fopen(filename, "w");

		for (d = 0; d < model->D; d++){

			doc = &(corpus->docs[d]);

			for (j = 0; j < model->m; j++){
				fprintf(fptheta, "%.10lf ", doc->theta[j]);
			}
			fprintf(fptheta, "\n");

			for (s = 0; s < doc->length; s++){

				sent = &(doc->sents[s]);

				for (c = 0; c < model->c; c++){
					if (c < model->c-1){
						fprintf(fpy, "%5.5lf ", (sent->py[c]));
					}
					else{
						fprintf(fpy, "%5.5lf |", (sent->py[c]));
					}
				}

			}
			fprintf(fpy, "\n");
			for (c = 0; c < model->c; c++){
				fprintf(fpb, "%5.5lf ", (doc->pb[c]));
			}
			fprintf(fpb, "\n");
		}
		fclose(fptheta);
		fclose(fpy);
		fclose(fpb);

	//}
	//save tau, gbar, hbar
	/*sprintf(filename, "%s.rho_beta", root);
	fileptr = fopen(filename, "w");
	for (j = 0; j < model->m; j++){
		fprintf(fileptr, "%5.10lf ", var->tau[j]);
	}
	fprintf(fileptr, "\n");
	for (j = 0; j < model->m; j++){
		fprintf(fileptr, "%5.10lf ", var->hbar[j]);
	}
	fprintf(fileptr, "\n");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%5.10lf ", var->gbar[j][n]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);*/


	//save adam_m and adam_v
	/*sprintf(filename, "%s.rho_w", root);
	fileptr = fopen(filename, "w");
	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%5.10lf ", var->adam_m[c][j]);
		}
		fprintf(fileptr, "\n");
	}
	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%5.10lf ", var->adam_v[c][j]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);*/


}


void load_model_continue(mltm_model* model, mltm_corpus* corpus, mltm_ss* ss,
		mltm_var* var, char* model_root){

	char filename[100];
	FILE* fileptr;
	int j, n, c, d;
	//double normsum;
	//sentence* sent;
	//float x;

	sprintf(filename, "%s.psi", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%lf %lf\n", &model->psi1, &model->psi2);
	fclose(fileptr);

	sprintf(filename, "%s.alpha", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%lf %lf\n", &model->alphahat, &model->alpha);
	fscanf(fileptr, "%Lf %Lf", &var->adam_m_alphahat, &var->adam_v_alphahat);
	fclose(fileptr);

	sprintf(filename, "%s.nu", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%lf %lf\n", &model->nuhat, &model->nu);
	fscanf(fileptr, "%Lf %Lf", &var->adam_m_nuhat, &var->adam_v_nuhat);
	fclose(fileptr);

	sprintf(filename, "%s.beta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			/*fscanf(fileptr, " %lf", &model->Elogbeta[j][n]);
			model->expElogbeta[j][n] = exp(model->Elogbeta[j][n]);
			var->mu[j][n] = model->nu + model->expElogbeta[j][n];
			var->summu[j] += var->mu[j][n];*/
			fscanf(fileptr, " %lf", &var->mu[j][n]);
		}
	}
	fclose(fileptr);

	sprintf(filename, "%s.w", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			fscanf(fileptr, "%lf ", &model->mu[c][j]);
			//model->muhat[c][j] = log(model->mu[c][j]/(1-model->mu[c][j]));
		}
	}
	fclose(fileptr);


	sprintf(filename, "%s.theta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (d = 0; d < corpus->ndocs; d++){

		for (j = 0; j < model->m; j++){
			fscanf(fileptr, "%lf ", &corpus->docs[d].theta[j]);
		}

	}
	fclose(fileptr);

	sprintf(filename, "%s.b", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (d = 0; d < corpus->ndocs; d++){

		for (c = 0; c < model->c; c++){
			fscanf(fileptr, "%lf ", &corpus->docs[d].pb[c]);
		}
	}
	fclose(fileptr);

	//load tau, gbar, hbar
	sprintf(filename, "%s.rho_beta", model_root);
	fileptr = fopen(filename, "r");
	for (j = 0; j < model->m; j++){
		fscanf(fileptr, "%lf ", &var->tau[j]);
	}
	for (j = 0; j < model->m; j++){
		fscanf(fileptr, "%lf ", &var->hbar[j]);
	}
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			fscanf(fileptr, "%lf ", &var->gbar[j][n]);
		}
	}
	fclose(fileptr);


	//save tau, gbar, hbar
	sprintf(filename, "%s.rho_w", model_root);
	fileptr = fopen(filename, "r");
	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			fscanf(fileptr, "%lf ", &var->adam_m[c][j]);
		}
	}
	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			fscanf(fileptr, "%lf ", &var->adam_v[c][j]);
		}
	}
	fclose(fileptr);

}


void load_model(mltm_model* model, mltm_corpus* corpus, mltm_ss* ss,
		mltm_var* var, char* model_root){

	char filename[100];
	FILE* fileptr;
	int j, n, c, d, i, s;
	double normsum;
	sentence* sent;
	//float x;
	sprintf(filename, "%s.psi", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%lf %lf\n", &model->psi1, &model->psi2);
	fclose(fileptr);

	sprintf(filename, "%s.alpha", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%lf %lf\n", &model->alphahat, &model->alpha);
	fscanf(fileptr, "%Lf %Lf", &var->adam_m_alphahat, &var->adam_v_alphahat);
	fclose(fileptr);

	sprintf(filename, "%s.nu", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%lf %lf\n", &model->nuhat, &model->nu);
	fscanf(fileptr, "%Lf %Lf", &var->adam_m_nuhat, &var->adam_v_nuhat);
	fclose(fileptr);

	sprintf(filename, "%s.beta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			/*fscanf(fileptr, " %lf", &model->Elogbeta[j][n]);
			model->expElogbeta[j][n] = exp(model->Elogbeta[j][n]);
			var->mu[j][n] = model->nu + model->expElogbeta[j][n];
			var->summu[j] += var->mu[j][n];*/
			fscanf(fileptr, " %lf", &var->mu[j][n]);
		}
	}
	fclose(fileptr);

	sprintf(filename, "%s.w", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			fscanf(fileptr, "%lf ", &model->mu[c][j]);
			//model->muhat[c][j] = log(model->mu[c][j]/(1-model->mu[c][j]));
		}
	}
	fclose(fileptr);

	for (d = 0; d < corpus->ndocs; d++){
		normsum = 0.0;
		for (j = 0; j < model->m; j++){
			corpus->docs[d].theta[j] = 1.0;//gsl_ran_flat(r, 0, 1);
			normsum += corpus->docs[d].theta[j];
		}
		for (j = 0; j < model->m; j++){
			corpus->docs[d].theta[j] /= normsum;
			corpus->docs[d].logtheta[j] = log(corpus->docs[d].theta[j]);
		}

		//initialize phi
		for (s = 0; s < corpus->docs[d].length; s++){
			sent = &(corpus->docs[d].sents[s]);
			for (i = 0; i < sent->length; i++){
				for (j = 0; j < model->m; j++){
					sent->phi[i][j] = 1.0/((double)model->m);
				}
			}
		}
	}

	//return(model);
}


void random_initialize_model(mltm_corpus* corpus, mltm_model * model, mltm_ss* ss, mltm_var* var){

	int d, n, j, c, i, s, chk, jj=0, ii;
	double u, cdf;
	double normsum;
	sentence* sent;
	document* doc;

	//Pinv = 1.0/model->p;

	//init sample beta, mu
	for (j = 0; j < model->m; j++){
		normsum = 0.0;
		for (n = 0; n < model->n; n++){
			var->mu[j][n] = 1.0 + model->nu + gsl_ran_flat(&(rng[0]), model->nu-0.1, model->nu+0.1);//gsl_ran_flat(&(rng[0]), 0, 100.0);
			normsum += var->mu[j][n];
		}

	}

	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			model->mu[c][j] = gsl_ran_flat(&(rng[0]), 0.5-.1, 0.5+.1);
			//model->mu[c][j] = exp(model->muhat[c][j])/(1+exp(model->muhat[c][j]));
		}
	}

	for (d = 0; d < corpus->ndocs; d++){
		normsum = 0.0;
		doc = &(corpus->docs[d]);
		for (j = 0; j < model->m; j++){
			doc->theta[j] = 1.0;//gsl_ran_flat(&(r[0]), 0, 1);
			normsum += doc->theta[j];
			doc->m[j] = 0.0;
		}
		doc->mbar = 0.0;
		for (j = 0; j < model->m; j++){
			doc->theta[j] /= normsum;
			doc->logtheta[j] = log(doc->theta[j]);
		}

		for (c = 0; c < model->c; c++){
			doc->maxy[c] = -1;
			doc->argmaxy[c] = 0;
		}
		//initialize phi
		for (s = 0; s < doc->length; s++){

			sent = &(doc->sents[s]);
			//Lsinv = 1.0/((double)sent->length);

			for (i = 0; i < sent->length; i++){
				u = gsl_ran_flat(&(rng[0]), 0, 1);
				chk = 0;
				cdf = 0.0;
				for (j = 0; j < model->m; j++){
					sent->phi[i][j] = 1.0/((double)model->m);

					if (chk == 0){
						//if (j > 0)	var->cdfphi[j] = var->cdfphi[j-1] + sent->phi[i][j];
						//else var->cdfphi[j] = sent->phi[i][j];
						cdf += sent->phi[i][j];
						if (u <= cdf){
							chk = 1;
							jj = j;
						}
					}
				}
				if (chk!=1){
					printf("%lf %lf\n", u, cdf);
					assert(0);
				}
				if ((jj >= model->m) || (jj < 0)){
					printf("%d\n", jj);
					assert(0);
				}
				sent->samples[i] = jj;
				doc->m[jj] += 1.0;
				doc->mbar += 1.0;
			}
			for (c = 0; c < model->c; c++){
				u = gsl_ran_flat(&(rng[0]), 0, 1);
				cdf = 0.0;
				chk = 0;
				for (i = 0; i < sent->length; i++){
					sent->x[c][i] = 1.0/((double)sent->length);
					cdf += sent->x[c][i];
					if (chk == 0){
						if (u <= cdf){
							chk = 1;
							ii = i;
						}
					}
				}
				sent->xsamples[c] = ii;
				jj = sent->samples[ii];
				sent->py[c] = model->mu[c][jj];

				if (sent->py[c] > doc->maxy[c]){
					doc->maxy[c] = sent->py[c];
					doc->argmaxy[c] = s;
				}
			}
		}

		for (j = 0; j < model->m; j++){
			var->gamma[j] = 0;
		}
		for (s = 0; s < doc->length; s++){
			for (i = 0; i < doc->sents[s].length; i++){
				var->gamma[doc->sents[s].samples[i]] += 1.0;
				if (doc->sents[s].samples[i] > model->m)
					printf("ohoay\n");
			}
		}
		for (j = 0; j < model->m; j++){
			if (var->gamma[j] != doc->m[j]){
				printf("%lf %lf\n", var->gamma[j], doc->m[j]);
				assert(0);
			}
		}
		for (c = 0; c < model->c; c++){
			doc->logpb[c] = log(doc->maxy[c]);
			doc->logp1mb[c] = log(1-doc->maxy[c]);
		}

	}

}

void corpus_initialize_model(mltm_corpus* corpus, mltm_model * model, mltm_ss* ss, mltm_var* var){

	int n, j, d, i, s, c, cnt, jj, chk, ii;
	double u, cdf, maxval;
	double normsum, temp;//, Lsinv, Pinv, temp;
	sentence* sent;
	document* doc;
	//Pinv = 1.0/model->p;

	int* docs = malloc(sizeof(int)*corpus->ndocs);
	for (d = 0; d < corpus->ndocs; d++){
		docs[d] = -1;
	}

	for (j = 0; j < model->m; j++){
		var->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			var->mu[j][n] = model->nu;
			var->summu[j] += var->mu[j][n];
		}

		for (i = 0; i < 40; i++){
			//choose a doc
			cnt = 0;
			do{
				d = floor(gsl_rng_uniform(&(rng[0]))*corpus->ndocs);
				if ((docs[d] == -1) || (cnt > 100)){
					docs[d] = j;
					break;
				}
				cnt += 1;
			}while(1);
			for (s = 0; s < corpus->docs[d].length; s++){
				sent = &(corpus->docs[d].sents[s]);
				for (n = 0; n < sent->length; n++){
					var->mu[j][sent->words[n]] += 1.0;
					var->summu[j] += 1.0;
				}
			}
		}
		for (n = 0; n < model->n; n++){
			var->mu[j][n] *= ((double)corpus->ndocs)/(var->summu[j]*40.0);
		}
		var->summu[j] = (double)corpus->ndocs/(40.0);

	}


	for (j = 0; j < model->m; j++){
		var->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			var->summu[j] += var->mu[j][n];
		}
		temp = gsl_sf_psi(var->summu[j]);
		for (n = 0; n < model->n; n++){
			//y = var->mu[j][n]/var->summu[j];
			//if (y == 0) y = 1e-50;
			//model->expElogbeta[j][n] = y;
			model->Elogbeta[n*model->m+j] = gsl_sf_psi(var->mu[j][n])-temp;
		}
	}


	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			model->mu[c][j] = gsl_ran_flat(&(rng[0]), 0.5-0.1, 0.5+0.1);
			//model->mu[c][j] = exp(model->muhat[c][j])/(1+exp(model->muhat[c][j]));
		}
	}


	for (d = 0; d < corpus->ndocs; d++){
		normsum = 0.0;
		doc = &(corpus->docs[d]);
		for (j = 0; j < model->m; j++){
			doc->theta[j] = 1.0;//gsl_ran_flat(&(r[0]), 0, 1);
			normsum += doc->theta[j];
			doc->m[j] = 0.0;
		}
		doc->mbar = 0.0;
		for (j = 0; j < model->m; j++){
			doc->theta[j] /= normsum;
			doc->logtheta[j] = log(doc->theta[j]);
		}
		for (c = 0; c < model->c; c++){
			doc->maxy[c] = -1;
			doc->argmaxy[c] = 0;
		}
		//initialize phi
		for (s = 0; s < doc->length; s++){
			sent = &(doc->sents[s]);
			//Lsinv = 1.0/((double)sent->length);

			for (i = 0; i < sent->length; i++){
				maxval = -1e200;
				for (j = 0; j < model->m; j++){
					sent->phi[i][j] = model->Elogbeta[sent->words[i]*model->m+j];
					if (sent->phi[i][j] > maxval)
						maxval = sent->phi[i][j];
				}
				normsum = 0.0;
				for (j = 0; j < model->m; j++){
					sent->phi[i][j] = model->alpha + exp(sent->phi[i][j] - maxval);
					//sent->phi[i][j] = 0.1 + exp(sent->phi[i][j] - maxval);
					normsum += sent->phi[i][j];
				}
				u = gsl_ran_flat(&(rng[0]), 0, 1);
				chk = 0;
				cdf = 0.0;
				for (j = 0; j < model->m; j++){
					//sent->phi[i][j] = 1.0/((double)model->m);
					sent->phi[i][j] /= normsum;

					if (chk == 0){
						//if (j > 0)	var->cdfphi[j] = var->cdfphi[j-1] + sent->phi[i][j];
						//else var->cdfphi[j] = sent->phi[i][j];
						cdf += sent->phi[i][j];
						if (u <= cdf){
							chk = 1;
							jj = j;
						}
					}
				}
				if (chk!=1){
					printf("%lf %lf\n", u, cdf);
					assert(0);
				}
				if ((jj >= model->m) || (jj < 0)){
					printf("%d\n", jj);
					assert(0);
				}
				sent->samples[i] = jj;
				doc->m[jj] += 1.0;
				doc->mbar += 1.0;
			}
			for (c = 0; c < model->c; c++){
				u = gsl_ran_flat(&(rng[0]), 0, 1);
				cdf = 0.0;
				chk = 0;
				for (i = 0; i < sent->length; i++){
					sent->x[c][i] = 1.0/((double)sent->length);
					cdf += sent->x[c][i];
					if (chk == 0){
						if (u <= cdf){
							chk = 1;
							ii = i;
						}
					}
				}
				sent->xsamples[c] = ii;
				jj = sent->samples[ii];
				sent->py[c] = model->mu[c][jj];

				if (sent->py[c] > doc->maxy[c]){
					doc->maxy[c] = sent->py[c];
					doc->argmaxy[c] = s;
				}
			}

		}
		for (j = 0; j < model->m; j++){
			var->gamma[j] = 0;
		}
		for (s = 0; s < doc->length; s++){
			for (i = 0; i < doc->sents[s].length; i++){
				var->gamma[doc->sents[s].samples[i]] += 1.0;
				if (doc->sents[s].samples[i] > model->m)
					printf("ohoay\n");
			}
		}
		for (j = 0; j < model->m; j++){
			if (var->gamma[j] != doc->m[j]){
				printf("%lf %lf\n", var->gamma[j], doc->m[j]);
				assert(0);
			}
		}
		for (c = 0; c < model->c; c++){
			doc->logpb[c] = log(doc->maxy[c]);
			doc->logp1mb[c] = log(1-doc->maxy[c]);
		}

	}

  	free(docs);
}

void loadtopics_initialize_model(mltm_corpus* corpus, mltm_model * model,
		mltm_ss* ss, mltm_var* var, char* model_root){

	int n, j, d, i, s, c, jj, chk, ii;
	double u, cdf, maxval;
	double normsum, temp;
	sentence* sent;
	document* doc;
	FILE* fileptr;
	char filename[400];

	//Pinv = 1.0/model->p;

	sprintf(filename, "%s.beta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			/*fscanf(fileptr, " %lf", &model->Elogbeta[j][n]);
			model->expElogbeta[j][n] = exp(model->Elogbeta[j][n]);
			var->mu[j][n] = model->nu + model->expElogbeta[j][n];
			var->summu[j] += var->mu[j][n];*/
			fscanf(fileptr, " %lf", &var->mu[j][n]);
		}
	}
	fclose(fileptr);

	for (j = 0; j < model->m; j++){
		var->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			var->mu[j][n] += model->nu;
			var->summu[j] += var->mu[j][n];
		}
		temp = gsl_sf_psi(var->summu[j]);
		for (n = 0; n < model->n; n++){
			//y = var->mu[j][n]/var->summu[j];
			//if (y == 0) y = 1e-50;
			//model->expElogbeta[j][n] = y;
			model->Elogbeta[n*model->m+j] = gsl_sf_psi(var->mu[j][n])-temp;
		}
	}


	sprintf(filename, "%s.w", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			fscanf(fileptr, "%lf ", &model->mu[c][j]);
			//model->muhat[c][j] = log(model->mu[c][j]/(1-model->mu[c][j]));
		}
	}
	fclose(fileptr);


	/*for (c = 0; c < model->c; c++){
		for (j = 0; j < model->m; j++){
			model->muhat[c][j] = 0.0;//gsl_ran_flat(&(rng[0]), -.1, .1);
			model->mu[c][j] = exp(model->muhat[c][j])/(1+exp(model->muhat[c][j]));
		}
	}*/


	for (d = 0; d < corpus->ndocs; d++){
		normsum = 0.0;
		doc = &(corpus->docs[d]);
		for (j = 0; j < model->m; j++){
			doc->theta[j] = 1.0;//gsl_ran_flat(&(r[0]), 0, 1);
			normsum += doc->theta[j];
			doc->m[j] = 0.0;
		}
		doc->mbar = 0.0;
		for (j = 0; j < model->m; j++){
			doc->theta[j] /= normsum;
			doc->logtheta[j] = log(doc->theta[j]);
		}
		for (c = 0; c < model->c; c++){
			doc->maxy[c] = -1;
			doc->argmaxy[c] = 0;
		}
		//initialize phi
		for (s = 0; s < doc->length; s++){
			sent = &(doc->sents[s]);
			//Lsinv = 1.0/((double)sent->length);

			for (i = 0; i < sent->length; i++){
				maxval = -1e200;
				for (j = 0; j < model->m; j++){
					sent->phi[i][j] = model->Elogbeta[sent->words[i]*model->m+j];
					if (sent->phi[i][j] > maxval)
						maxval = sent->phi[i][j];
				}
				normsum = 0.0;
				for (j = 0; j < model->m; j++){
					sent->phi[i][j] = model->alpha + exp(sent->phi[i][j] - maxval);
					normsum += sent->phi[i][j];
				}
				u = gsl_ran_flat(&(rng[0]), 0, 1);
				chk = 0;
				cdf = 0.0;
				for (j = 0; j < model->m; j++){
					//sent->phi[i][j] = 1.0/((double)model->m);
					sent->phi[i][j] /= normsum;

					if (chk == 0){
						//if (j > 0)	var->cdfphi[j] = var->cdfphi[j-1] + sent->phi[i][j];
						//else var->cdfphi[j] = sent->phi[i][j];
						cdf += sent->phi[i][j];
						if (u <= cdf){
							chk = 1;
							jj = j;
						}
					}
				}
				if (chk!=1){
					printf("%lf %lf\n", u, cdf);
					assert(0);
				}
				if ((jj >= model->m) || (jj < 0)){
					printf("%d\n", jj);
					assert(0);
				}
				sent->samples[i] = jj;
				doc->m[jj] += 1.0;
				doc->mbar += 1.0;
			}
			for (c = 0; c < model->c; c++){
				u = gsl_ran_flat(&(rng[0]), 0, 1);
				cdf = 0.0;
				chk = 0;
				for (i = 0; i < sent->length; i++){
					sent->x[c][i] = 1.0/((double)sent->length);
					cdf += sent->x[c][i];
					if (chk == 0){
						if (u <= cdf){
							chk = 1;
							ii = i;
						}
					}
				}
				sent->xsamples[c] = ii;
				jj = sent->samples[ii];
				sent->py[c] = model->mu[c][jj];

				if (sent->py[c] > doc->maxy[c]){
					doc->maxy[c] = sent->py[c];
					doc->argmaxy[c] = s;
				}
			}
		}
		for (j = 0; j < model->m; j++){
			var->gamma[j] = 0;
		}
		for (s = 0; s < doc->length; s++){
			for (i = 0; i < doc->sents[s].length; i++){
				var->gamma[doc->sents[s].samples[i]] += 1.0;
				if (doc->sents[s].samples[i] > model->m)
					printf("ohoay\n");
			}
		}
		for (j = 0; j < model->m; j++){
			if (var->gamma[j] != doc->m[j]){
				printf("%lf %lf\n", var->gamma[j], doc->m[j]);
				assert(0);
			}
		}
		for (c = 0; c < model->c; c++){
			doc->logpb[c] = log(doc->maxy[c]);
			doc->logp1mb[c] = log(1-doc->maxy[c]);
		}

	}


}


double log_sum(double log_a, double log_b)
{
	double v;

	if (log_a < log_b){
		v = log_b+log(1 + exp(log_a-log_b));
	}
	else{
		v = log_a+log(1 + exp(log_b-log_a));
	}
	return(v);
}

double log_subtract(double log_a, double log_b)
{
	double v;

	if (log_a < log_b){
		v = log_b+log(1 - exp(log_a-log_b));
	}
	else{
		v = log_a+log(1 - exp(log_b-log_a));
	}
	return(v);
}

void random_permute(int size, int* vec){

	int i, j, temp;
	
	for (j = 0; j < size; j++){
		vec[j] = j;
	}
	for (i = size-1; i > 0; i--){
		j = ((int)(size*gsl_rng_uniform(&(rng[0]))))%(i+1);
		temp = vec[j];
		vec[j] = vec[i];
		vec[i] = temp;
	}

}
