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
	//int num_classes, num_topics;
	//double sigma2;
	long int seed;


	seed = atoi(argv[1]);

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r, seed);

	printf("SEED = %ld\n", seed);

	MAXITER = 5000;
	CONVERGED = 1e-4;
	NUMINIT = 10;
	BURNIN = 1000;

	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);

	if (argc > 1){
		if (strcmp(task, "train")==0){
			strcpy(label_file,argv[4]);
			strcpy(settings_file,argv[5]);
			strcpy(init,argv[6]);
			strcpy(dir,argv[7]);
			if (strcmp(init, "load") == 0)
				strcpy(model_name, argv[8]);
			train(corpus_file, label_file, settings_file, init, dir, model_name);

			gsl_rng_free (r);
			return(0);
		}
		/*if (strcmp(task, "test")==0){ // should run lda for test
			strcpy(settings_file,argv[4]);
			strcpy(model_name,argv[5]);
			strcpy(dir,argv[6]);
			test(corpus_file, settings_file, model_name, dir);

			gsl_rng_free (r);
			return(0);
		}*/
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
	int iteration, ntopics, nclasses, ndocs;
	double doclkh, wrdlkh;
	double normsum, temp, alpha_sigma, nu_sigma;
	int d, n, j, i;

	mltm_corpus* corpus;
	mltm_model *model = NULL;
	mltm_mcmc *mcmc = NULL;
	mltm_var *var = NULL;

	time_t t1,t2;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "general %d\n", &i);
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &ndocs);
	fscanf(fp, "N %d\n", &n);
	fscanf(fp, "T %d\n", &MAXITER);
	fscanf(fp, "BURNIN %d\n", &BURNIN);
	fscanf(fp, "alpha %lf\n", &temp);
	fscanf(fp, "nu %lf\n", &temp);
	fscanf(fp, "psi %lf %lf\n", &temp, &temp);
	fscanf(fp, "alpha_sigma %lf\n", &alpha_sigma);
	fscanf(fp, "nu_sigma %lf\n", &nu_sigma);
	fclose(fp);

	corpus = read_data(dataset, nclasses, ndocs, 1, lblfile, ntopics);
	model = new_mltm_model(corpus, settings_file);
	mcmc = new_mltm_mcmc(model);


	corpus->nterms = model->n;
	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file
	sprintf(string, "%s/likelihood.dat", dir);
	lhood_fptr = fopen(string, "w");

	if (strcmp(start, "random")==0){
	printf("random\n");
	random_initialize_model(corpus, model, mcmc);
	}
	/*else if (strcmp(start, "load")==0){
	printf("load\n");
	model = new_mltm_model(ntopics, nclasses, corpus->nterms, sigma2);
		model->D = corpus->ndocs;
		ss = new_mltm_ss(model);
		var =  new_mltm_var(model, nmax);

		random_initialize_model(model, corpus, ss, var);

		//load beta
		sprintf(filename, "%s.beta", model_name);
		printf("loading %s\n", filename);
		fileptr = fopen(filename, "r");
		for (n = 0; n < model->n; n++){
			for (j = 0; j < ntopics; j++){
				fscanf(fileptr, " %lf", &y);
				model->logbeta[j][n] = y;
				model->beta[j][n] = exp(model->logbeta[j][n]);
			}
		}
		fclose(fileptr);

		//load alpha
		sprintf(filename, "%s.alpha", model_name);
		printf("loading %s\n", filename);
		fileptr = fopen(filename, "r");
		for (j = 0; j < ntopics; j++){
			for (c = 0; c < model->c; c++){
				fscanf(fileptr, " %lf", &y);
				model->alpha[j][c] = y + EPS;
			}
		}
		fclose(fileptr);

	}*/							
	int maxs = 0;
	int s, maxLs = 0;
	for (d = 0; d < model->D; d++){
		if (corpus->docs[d].length > maxs)	maxs = corpus->docs[d].length;
		for (s = 0; s < corpus->docs[d].length; s++){
			if (corpus->docs[d].sents[s].length > maxLs)	maxLs = corpus->docs[d].sents[s].length;
		}
	}

	var = malloc(sizeof(mltm_var));
	var->phi = malloc(sizeof(double)*model->m);
	var->gamma = malloc(sizeof(double)*maxLs);
	var->z = malloc(sizeof(unsigned int)*model->m);
	var->temp1 = malloc(sizeof(double)*model->m);
	var->temp1prime = malloc(sizeof(double)*model->m);
	var->temp2 = malloc(sizeof(double)*model->m);
	var->temp2prime = malloc(sizeof(double)*model->m);
	var->yprime = malloc(sizeof(int)*maxs); //max # sent
	var->yprob = malloc(sizeof(double)*maxs); //max # sent

	iteration = 0;
	//sprintf(filename, "%s/%03d", dir,iteration);
	//printf("%s\n",filename);
	//write_mltm_model(corpus, model, mcmc, filename);

	time(&t1);
	sprintf(filename, "%s/likelihood.dat", dir);
	lhood_fptr = fopen(filename, "w");

	int tt, c, jj;
	double alphaprime, nuprime, logP, logPprime, logH, temp_accRate, u, numj;
	double prev_nu, prev_alpha;
	double gt_lbls_not_used, temp_lbls_not_used, total_on_lbls;;
	gt_lbls_not_used = 0.0;
	mcmc->a = 1;
	wrdlkh = 0.0;
	mcmc->acceptrate_alpha = 0.0;
	mcmc->acceptrate_nu = 0.0;
	do{


		mcmc->n += 1.0;
		if (mcmc->n > model->BurnIn){
			if ((int)(mcmc->n-model->BurnIn)%mcmc->b == 0){
				mcmc->zybar2 = 0.0;
				mcmc->zhat2 = 0.0;
				mcmc->znum = 0.0;
			}
		}
		temp_lbls_not_used = 0.0;
		total_on_lbls = 0.0;

		for (d = 0; d < corpus->ndocs; d++){

			doc_mcmc(&(corpus->docs[d]), model, mcmc, d, var);

			//lhood += doclkh;
			for (c = 0; c < model->c; c++){
				if ((corpus->docs[d].b[c] == 1)){
					total_on_lbls += 1.0;
					if (corpus->docs[d].union_lbl[c] != 1)
						temp_lbls_not_used += 1.0;
				}
			}
		}
		temp_lbls_not_used /= total_on_lbls;
		if (mcmc->n > model->BurnIn){
			gt_lbls_not_used = ((mcmc->n-1.0-model->BurnIn)*gt_lbls_not_used +
                                        temp_lbls_not_used)/(mcmc->n-model->BurnIn);
		}

		//sample alpha and nu

		temp_accRate = 0;
		alphaprime = exp((gsl_ran_gaussian(r, alpha_sigma)) + log(model->alpha));
		logP = (model->psi0[0]-1)*log(model->alpha) - model->alpha/model->psi0[1];// +
				//model->D*lgamma(model->alpha*model->m)-model->D*model->m*lgamma(model->alpha);
		logPprime = (model->psi0[0]-1)*log(alphaprime) - alphaprime/model->psi0[1];// +
				//model->D*lgamma(alphaprime*model->m)-model->D*model->m*lgamma(alphaprime);
		for (d = 0; d < model->D; d++){
			temp = 0.0;
			numj = 0.0;
			for (c = 0; c < (model->c+model->general); c++){
				if ((c == model->c) || (corpus->docs[d].b[c] == 1)){
					for (jj = 0; jj < model->m_per_c; jj++){
						j = model->tpcs_of_lbl[c][jj];
						numj += 1.0;
						logP += lgamma(model->alpha+mcmc->m[j][d]);
						logPprime += lgamma(alphaprime+mcmc->m[j][d]);
						temp += mcmc->m[j][d];
					}
				}
			}
			logP += lgamma(model->alpha*numj) - numj*lgamma(model->alpha);
			logPprime += lgamma(alphaprime*numj) - numj*lgamma(alphaprime);

			logP -= lgamma(model->alpha*model->m+temp);
			logPprime -= lgamma(alphaprime*model->m+temp);
		}
		logH = logPprime - logP;
		u = log(gsl_ran_flat(r, 0,1));
		if (u < min(0, logH)){
			temp_accRate += 1.0;
			model->alpha = alphaprime;
			for (d = 0; d < model->D; d++){
				for (c = 0; c < (model->c+model->general); c++){
					if ((c == model->c) || (corpus->docs[d].b[c] == 1)){
						for (jj = 0; jj < model->m_per_c; jj++){
							j = model->tpcs_of_lbl[c][jj];
							mcmc->logm[j][d] = log(model->alpha + mcmc->m[j][d]);
						}
					}
				}
			}
		}
		if (mcmc->n > model->BurnIn){
			model->MCalpha = ((mcmc->n-1.0-model->BurnIn)*model->alpha +
					model->alpha)/(mcmc->n-model->BurnIn);
		}
		mcmc->acceptrate_alpha = ((mcmc->n-1)*mcmc->acceptrate_alpha + temp_accRate)/mcmc->n;


		//sample nu
		temp_accRate = 0;
		nuprime = exp((gsl_ran_gaussian(r, nu_sigma)) + log(model->nu));
		logP = (model->psi0[0]-1)*log(model->nu) - model->nu/model->psi0[1] +
				model->m*lgamma(model->nu*model->n)-model->m*model->n*lgamma(model->nu);
		logPprime = (model->psi0[0]-1)*log(nuprime) - nuprime/model->psi0[1] +
				model->m*lgamma(nuprime*model->n)-model->m*model->n*lgamma(nuprime);
		for (j = 0; j < model->m; j++){
			for (n = 0; n < model->n; n++){
				logP += lgamma(model->nu+mcmc->t[j][n]);
				logPprime += lgamma(nuprime+mcmc->t[j][n]);
			}
			logP -= lgamma(model->nu*model->n+mcmc->tbar[j]);
			logPprime -= lgamma(nuprime*model->n+mcmc->tbar[j]);
		}
		logH = logPprime - logP;
		u = log(gsl_ran_flat(r, 0,1));
		if (u < min(0, logH)){
			temp_accRate += 1.0;
			model->nu = nuprime;
			//printf("%lf\n", model->nu);
			for (j = 0; j < model->m; j++){
				for (n = 0; n < model->n; n++){
					mcmc->logt[j][n] = log(model->nu+mcmc->t[j][n]);
				}
				mcmc->logtbar[j] = log(model->nu*model->n+mcmc->tbar[j]);
			}
		}
		if (mcmc->n > model->BurnIn){
			model->MCnu = ((mcmc->n-1.0-model->BurnIn)*model->nu +
					model->nu)/(mcmc->n-model->BurnIn);
		}
		mcmc->acceptrate_nu = ((mcmc->n-1)*mcmc->acceptrate_nu+ temp_accRate)/mcmc->n;
		


		if (mcmc->n > model->BurnIn){
			tt = (int)(mcmc->n-model->BurnIn)%mcmc->b;

			// update posterior of beta and mu
			for (j = 0; j < model->m; j++){
				normsum = model->nu*model->n + mcmc->tbar[j];
				for (n = 0; n < model->n; n++){
					model->MCbeta[j][n] = ((mcmc->n-1.0-model->BurnIn)*model->MCbeta[j][n] +
							(model->nu+mcmc->t[j][n])/normsum)/(mcmc->n-model->BurnIn);
				}

			}


			if (tt == 0){
				mcmc->zybar2 /= mcmc->znum;
				mcmc->zhat2 /= mcmc->znum;
				mcmc->zmeanybar2 = ((mcmc->a-1.0)*mcmc->zmeanybar2 + mcmc->zybar2)/mcmc->a;
				mcmc->mcse_z = mcmc->b*(mcmc->zmeanybar2 - mcmc->zhat2);
				mcmc->mcse_z = sqrt(mcmc->mcse_z/(mcmc->n-model->BurnIn));
				//printf("%lf %lf, %lf\n", mcmc->zmeanybar2, mcmc->zhat2, mcmc->mcse_z);
				mcmc->a += 1;
			}
		}

		if (iteration%100 == 0){
			//compute likelihood
			wrdlkh = 0.0;
			doclkh = 0.0;
			for (d = 0; d < model->D; d++){
				doclkh = 0.0;
				for (s = 0; s < corpus->docs[d].length; s++){
					for (i = 0; i < corpus->docs[d].sents[s].length; i++){
						temp = 0.0;
						for (j = 0; j < model->m; j++)
							temp += model->MCtheta[j][d]*model->MCbeta[j][corpus->docs[d].sents[s].words[i]];
						doclkh += log(temp);
					}
				}
				wrdlkh += doclkh;
			}
			sprintf(filename, "%s/%03d", dir,1);
			write_mltm_model(corpus, model, mcmc, filename);
			time(&t2);
			fprintf(lhood_fptr, "%d %e %5ld %lf %lf %lf %lf %lf\n",
					iteration, wrdlkh, (int)t2-t1, mcmc->mcse_z, mcmc->acceptrate_alpha,
					mcmc->acceptrate_nu, temp_lbls_not_used, gt_lbls_not_used);
			fflush(lhood_fptr);
			printf("***** MCMC ITERATION %d *****, MCStdEr_z = %lf\n",
					iteration, mcmc->mcse_z);
		}

		iteration ++;

		if ((mcmc->n > MAXITER) || ((mcmc->n > (model->BurnIn+1000)) && (mcmc->mcse_z < 0.02)))
			break;

	}while(1);

	fclose(lhood_fptr);

	sprintf(filename, "%s/final", dir);

	write_mltm_model(corpus, model, mcmc, filename);

}



void doc_mcmc(document* doc, mltm_model* model, mltm_mcmc* mcmc, int d, mltm_var* var){

	int s, i, j, c, n, tt, ind, jj, cc;
	int prevz, check;
	double maxval, normsum, u;
	sentence* sent;

	tt = (int)(mcmc->n-model->BurnIn)%mcmc->b;

	for (c = 0; c < model->c; c++){
		doc->union_lbl[c] = 0;
	}
	for (s = 0; s < doc->length; s++){

		sent = &(doc->sents[s]);

		for (i = 0; i < sent->length; i++){

			n = sent->words[i];

			//sample z
			prevz = sent->z[i];
			mcmc->m[prevz][d] -= 1.0;
			mcmc->logm[prevz][d] = log(model->alpha + mcmc->m[prevz][d]);

			mcmc->t[prevz][n] -= 1.0;
			mcmc->logt[prevz][n] = log(model->nu + mcmc->t[prevz][n]);
			mcmc->tbar[prevz] -= 1.0;
			mcmc->logtbar[prevz] = log(model->nu*model->n + mcmc->tbar[prevz]);

			ind = 0;
			maxval = -1e50;
			for (c = 0; c < (model->c+model->general); c++){
				if ((c == model->c) || (doc->b[c] == 1)){
					for (jj = 0; jj < model->m_per_c; jj++){
						j = model->tpcs_of_lbl[c][jj];
						var->phi[ind] = mcmc->logt[j][n] - mcmc->logtbar[j] + mcmc->logm[j][d];
							//var->phi[ind] = log((model->nu + mcmc->t[j][n])/(model->nu*model->n + mcmc->tbar[j]))
							//	+ log(model->alpha + mcmc->m[j][d]);

						var->z[ind] = j;
						if (var->phi[j] > maxval)	maxval = var->phi[j];
						ind += 1;
					}
				}
			}
			normsum = 0.0;
			for (jj = 0; jj < ind; jj++){
				var->phi[jj] = exp(var->phi[jj] - maxval);
				normsum += var->phi[jj];
			}
			var->phi[0] /= normsum;
			for (jj = 1; jj < ind; jj++){
				var->phi[jj] = var->phi[jj-1] + var->phi[jj]/normsum;
			}

			u = gsl_ran_flat(r, 0, 1);
			check = 0;
			for (jj = 0; jj < ind; jj++){
				if (u <= var->phi[jj]){
					check = 1;
					break;
				}
			}
			if (check == 0) printf("Logical Error: doc = %d, sent = %d, class = %c, Ls = %d, phi[0] = %lf\n",
					d,s,c,sent->length, var->phi[0]);

			j = var->z[jj];
			sent->z[i] = j;
			cc = model->lbl_of_tpc[j];
			if (cc < model->c) //it's not one of general topics
				doc->union_lbl[cc] = 1;
			if (mcmc->n > model->BurnIn){
				sent->MCz[i] = ((mcmc->n-1.0-model->BurnIn)*sent->MCz[i] +
						(double)j)/(mcmc->n-model->BurnIn);
				sent->MClblz[i] = ((mcmc->n-1.0-model->BurnIn)*sent->MClblz[i] +
						model->lbl_of_tpc[j])/(mcmc->n-model->BurnIn);

				if (tt > 0)
					sent->YbarMCz[i] = ((tt-1.0)*sent->YbarMCz[i] +
							(double)j)/tt;
				else{
					sent->YbarMCz[i] = ((mcmc->b-1.0)*sent->YbarMCz[i] +
							(double)j)/mcmc->b;
					mcmc->zybar2 += pow(sent->YbarMCz[i], 2.0);
					mcmc->zhat2 += pow(sent->MCz[i], 2.0);
					mcmc->znum += 1.0;

				}
				//sent->MCzvec[i][j] += 1.0;
			}
			mcmc->m[j][d] += 1.0;
			mcmc->logm[j][d] = log(model->alpha + mcmc->m[j][d]);
			mcmc->t[j][n] += 1.0;
			mcmc->logt[j][n] = log(model->nu + mcmc->t[j][n]);
			mcmc->tbar[j] += 1.0;
			mcmc->logtbar[j] = log(model->nu*model->n + mcmc->tbar[j]);


		}

	}


	//update posterior estimate of theta
	normsum = 0.0;
	for (j = 0; j < model->m; j++){
		normsum += model->alpha + mcmc->m[j][d];
	}
	if (mcmc->n > model->BurnIn){
		for (j = 0; j < model->m; j++){
			model->MCtheta[j][d] = ((mcmc->n-1.0-model->BurnIn)*model->MCtheta[j][d] +
					(model->alpha+mcmc->m[j][d])/normsum)/(mcmc->n-model->BurnIn);
		}
	}

}



mltm_model* new_mltm_model(mltm_corpus* corpus, char* settings_file)
{
	int n, j, c, d;
	//double temp1, temp2;
	FILE* fp;

	mltm_model* model = malloc(sizeof(mltm_model));
	model->c = 0;
	model->m = 0;
	model->D = 0;
	model->n = 0;
	model->T = 0;
	model->alpha = 0.0;
	model->nu = 0.0;
	model->psi0 = malloc(sizeof(double)*2);

	fp = fopen(settings_file, "r");
	fscanf(fp, "general %d\n", &model->general);
	fscanf(fp, "M %d\n", &model->m);
	fscanf(fp, "C %d\n", &model->c);
	fscanf(fp, "D %d\n", &model->D);
	fscanf(fp, "N %d\n", &model->n);
	fscanf(fp, "T %d\n", &MAXITER);
	fscanf(fp, "BURNIN %d\n", &model->BurnIn);
	fscanf(fp, "alpha %lf\n", &model->alpha);
	fscanf(fp, "nu %lf\n", &model->nu);
	fscanf(fp, "psi %lf %lf\n", &model->psi0[0], &model->psi0[1]);
	fclose(fp);

	if (model->general==1)
		model->m_per_c = model->m/(model->c+1);
	else
		model->m_per_c = model->m/(model->c);

	model->MCbeta = malloc(sizeof(double*)*model->m);
	model->MCtheta = malloc(sizeof(double*)*model->m);
	model->lbl_of_tpc = malloc(sizeof(int)*model->m);
	for (j = 0; j < model->m; j++){
		model->lbl_of_tpc[j] = (int)(j-j%model->m_per_c)/model->m_per_c;
		printf("%d %d \n",j, model->lbl_of_tpc[j]);
		model->MCbeta[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			model->MCbeta[j][n] = 0.0;
		}
		model->MCtheta[j] = malloc(sizeof(double)*model->D);
		for (d = 0; d < model->D; d++){
			model->MCtheta[j][d] = 0.0;
		}
	}

	//convention: class C+1 is the general class
	model->tpcs_of_lbl = malloc(sizeof(int*)*(model->c+model->general));
	for (c = 0; c < (model->c+model->general); c++){
		model->tpcs_of_lbl[c] = malloc(sizeof(int)*model->m_per_c);
		for (j = 0; j < model->m_per_c; j++){
			model->tpcs_of_lbl[c][j] = j + c*model->m_per_c;
		}
	}

	return(model);
}

mltm_mcmc * new_mltm_mcmc(mltm_model* model){

	int n, j, d;

	mltm_mcmc* mcmc = malloc(sizeof(mltm_mcmc));

	mcmc->n = 0;
	mcmc->b = 100;

	mcmc->t = malloc(sizeof(double*)*model->m);
	mcmc->tbar = malloc(sizeof(double)*model->m);
	mcmc->m = malloc(sizeof(double*)*model->m);
	mcmc->logt = malloc(sizeof(double*)*model->m);
	mcmc->logtbar = malloc(sizeof(double)*model->m);
	mcmc->logm = malloc(sizeof(double*)*model->m);

	for (j = 0; j < model->m; j++){
		mcmc->tbar[j] = 0.0;
		mcmc->t[j] = malloc(sizeof(double)*model->n);
		mcmc->logtbar[j] = 0.0;
		mcmc->logt[j] = malloc(sizeof(double)*model->n);
		for(n = 0; n < model->n; n++){
			mcmc->t[j][n] = 0.0;
			mcmc->logt[j][n] = 0.0;
		}

		mcmc->m[j] = malloc(sizeof(double)*model->D);
		mcmc->logm[j] = malloc(sizeof(double)*model->D);
		for (d = 0; d < model->D; d++){
			mcmc->m[j][d] = 0.0;
			mcmc->logm[j][d] = 0.0;
		}

	}

	return(mcmc);
}




mltm_corpus* read_data(const char* data_filename, int nclasses, int ndocs,
		int lblchck, const char* lbl_filename, int ntopics)
{
	FILE *fileptr;
	int Ls, word, nd, nw, lbl;
	int Sd, s, i, c;
	mltm_corpus* corpus;

	printf("reading data from %s\n", data_filename);
	corpus = malloc(sizeof(mltm_corpus));
	//corpus->docs = 0;
	corpus->docs = (document *) malloc(sizeof(document)*(ndocs));
	corpus->nterms = 0.0;
	corpus->ndocs = 0.0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "<%10d> ", &Sd) != EOF)){ // number of sentences
		//corpus->docs = (document *) realloc(corpus->docs, sizeof(document)*(nd+1));
		corpus->docs[nd].sents = (sentence *) malloc(sizeof(sentence)*Sd);
		corpus->docs[nd].length = Sd;
		corpus->docs[nd].b = malloc(sizeof(int)*nclasses);
		corpus->docs[nd].union_lbl = malloc(sizeof(int)*nclasses);
		for (c = 0; c < nclasses; c++){
			corpus->docs[nd].b[c] = 0;
			corpus->docs[nd].union_lbl[c] = 0;
		}
		for (s = 0; s < Sd; s++){
			fscanf(fileptr, "<%10d> ", &Ls);
			corpus->docs[nd].sents[s].words = malloc(sizeof(int)*Ls);
			corpus->docs[nd].sents[s].length = Ls;
			corpus->docs[nd].sents[s].z = malloc(sizeof(int)*Ls);
			corpus->docs[nd].sents[s].MCz = malloc(sizeof(double)*Ls);
			corpus->docs[nd].sents[s].MClblz = malloc(sizeof(double)*Ls);
			corpus->docs[nd].sents[s].YbarMCz = malloc(sizeof(double)*Ls);
			for (i = 0; i < Ls; i++){
				fscanf(fileptr, "%10d ", &word);
				corpus->docs[nd].sents[s].words[i] = word;
				corpus->docs[nd].sents[s].YbarMCz[i] = 0.0;
				corpus->docs[nd].sents[s].MCz[i] = 0.0;
				/*corpus->docs[nd].sents[s].MCzvec[i] = malloc(sizeof(double)*ntopics);
				for (j = 0; j < ntopics; j++)
					corpus->docs[nd].sents[s].MCzvec[i][j] = 0.0;*/
				if (word >= nw) { nw = word + 1; }
			}
		}
		nd++;
	}
	fclose(fileptr);
	corpus->ndocs = nd;
	corpus->nterms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);
	//printf("doc %d, sent %d, word %d is %d\n", 1, 1, 1, corpus->docs[0].sents[0].words[0]);
	for (nd = 0; nd < corpus->ndocs; nd++){
		for (c = 0; c < nclasses; c++){
			corpus->docs[nd].b[c] = 0;
			//printf("doc %d, sent %d, word %d is %d\n", 1, 1, 1, corpus->docs[0].sents[0].words[0]);
		}
	}
	if (lblchck == 1){
		//printf("reading data from %s\n", lbl_filename);
		fileptr = fopen(lbl_filename, "r");
		for (nd = 0; nd < corpus->ndocs; nd++){
			for (c = 0; c < nclasses; c++){
				fscanf(fileptr, "%d", &lbl);
				corpus->docs[nd].b[c] = lbl;
				//printf("doc %d, sent %d, word %d is %d\n", 1, 1, 1, corpus->docs[0].sents[0].words[0]);
				//printf("%d %d %d\n", nd, c, lbl);
			}
		}
		fclose(fileptr);
	}
	//printf("doc %d, sent %d, word %d is %d\n", 1, 1, 1, corpus->docs[0].sents[0].words[0]);
	return(corpus);
}


void write_mltm_model(mltm_corpus * corpus, mltm_model * model, mltm_mcmc* mcmc, char * root)
{
	char filename[200];
	FILE* fileptr;
	int n, j, d, s, i;

	sprintf(filename, "%s.tpc_lbl", root);
	fileptr = fopen(filename, "w");
	for (j = 0; j < model->m; j++){
		fprintf(fileptr, "%d ", model->lbl_of_tpc[j]);
	}
	fclose(fileptr);


	//alpha
	sprintf(filename, "%s.alpha", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "%lf", model->MCalpha);
	fclose(fileptr);
	//nu
	sprintf(filename, "%s.nu", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "%lf", model->MCnu);
	fclose(fileptr);



	//beta
	sprintf(filename, "%s.beta", root);
	fileptr = fopen(filename, "w");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%.10lf ",model->MCbeta[j][n]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	//theta
	sprintf(filename, "%s.theta", root);
	fileptr = fopen(filename, "w");
	for (d = 0; d < model->D; d++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%.10lf ",model->MCtheta[j][d]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	//Y
	sprintf(filename, "%s.y", root);
	fileptr = fopen(filename, "w");
	for (d = 0; d < model->D; d++){
		for (s = 0; s < corpus->docs[d].length; s++){
			fprintf(fileptr, "<");
			for (i = 0; i < corpus->docs[d].sents[s].length; i++){
				if (i > 0)
					fprintf(fileptr, ",%2.2lf",corpus->docs[d].sents[s].MClblz[i]);
				else
					fprintf(fileptr, "%2.2lf",corpus->docs[d].sents[s].MClblz[i]);
			}
			fprintf(fileptr, ">");
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
}



void random_initialize_model(mltm_corpus* corpus, mltm_model * model, mltm_mcmc* mcmc){

	int i, s, d, n, j, c, ind, jj;
	int check;
	double u, normsum, maxval;

	double* theta = malloc(sizeof(double)*model->m);
	double* p = malloc(sizeof(double)*model->m);
	double* alpha = malloc(sizeof(double)*model->m);
	int* z = malloc(sizeof(int)*model->m);
	double* nu = malloc(sizeof(double)*model->n);
	double* beta = malloc(sizeof(double)*model->n);


	printf("doc %d, sent %d, word %d is %d\n", 5, 3, 2, corpus->docs[0].sents[0].words[0]);
	mcmc->n = 1;
	for (j = 0; j < model->m; j++){
		theta[j] = 0.0;
		z[j] = 0;
		p[j] = 0.0;
		alpha[j] = 0.0;
	}
	for (n = 0; n < model->n; n++){
		beta[n] = 0.0;
		nu[n] = 0.0;
	}

	//init sample beta, mu
	for (j = 0; j < model->m; j++){
		for (n = 0; n < model->n; n++){
			nu[n] = model->nu;
		}
		gsl_ran_dirichlet (r, model->n, nu, beta);
		for (n = 0; n < model->n; n++){
			model->MCbeta[j][n] = beta[n];
			if (model->MCbeta[j][n] == 0) model->MCbeta[j][n] += 1e-20;
		}

	}

	for (d = 0; d < corpus->ndocs; d++){
		//init sample theta
		for (j = 0; j < model->m; j++){
			alpha[j] = model->alpha;
		}
		gsl_ran_dirichlet (r, model->m, alpha, theta);

		for (j = 0; j < model->m; j++){
			model->MCtheta[j][d] = theta[j];
			if (model->MCtheta[j][d] == 0) model->MCtheta[j][d] += 1e-20;
		}
		//sample z
		for (s = 0; s < corpus->docs[d].length; s++){
			for (i = 0; i < corpus->docs[d].sents[s].length; i++){
				n = corpus->docs[d].sents[s].words[i];
				ind = 0;
				normsum = 0.0;
				maxval = -1e50;
				for (c = 0; c < (model->c+model->general); c++){
					if ((c == model->c) || (corpus->docs[d].b[c] == 1)){
						for (jj = 0; jj < model->m_per_c; jj++){
							j = model->tpcs_of_lbl[c][jj];
							//p[ind] = log(model->MCtheta[j][d])+log(model->MCbeta[j][n]);
							p[ind] = log(model->MCbeta[j][n]); //assuming theta init is uniform
							//if ((d == 508) && (i == 2) && (s==5))
							//	printf("**%d %d %f %f %f\n", ind,j,p[ind],(model->MCtheta[j][d]),(model->MCbeta[j][n]));
							z[ind] = j;
							if (p[ind] > maxval) maxval = p[ind];
							ind += 1;
						}
					}
				}
				normsum = 0.0;
				for (jj = 0; jj < ind; jj++){
					//if ((d == 508) && (s==5))
						//printf("%f %f %f\n", maxval, p[jj], exp(p[jj] - maxval));
					p[jj] = exp(p[jj] - maxval);
					normsum += p[jj];
				}
				p[0] /= normsum;
				for (jj = 1; jj < ind; jj++){
					p[jj] = p[jj-1] + p[jj]/normsum;
				}

				u = gsl_ran_flat(r, 0, 1);
				check = 0;
				for (jj = 0; jj < ind; jj++){
					if (u <= p[jj]){
						check = 1;
						break;
					}
				}
				if (check == 0) {
					printf("Logical Error: doc = %d, sent = %d, i = %d, Ls = %d, phi[0] = %lf\n",d,s,i,corpus->docs[d].sents[s].length, p[0]);
				}

				j = z[jj];
				corpus->docs[d].sents[s].z[i] = j;
				corpus->docs[d].sents[s].MCz[i] = (double)j;
				corpus->docs[d].sents[s].MClblz[i] = (double)model->lbl_of_tpc[j];
				mcmc->m[j][d] += 1.0;
				mcmc->t[j][n] += 1.0;
				mcmc->tbar[j] += 1.0;

			}

		}


		//update theta
		normsum = 0.0;
		for (j = 0; j < model->m; j++){
			model->MCtheta[j][d] = model->alpha + mcmc->m[j][d];
			normsum += model->MCtheta[j][d];
		}
		for (j = 0; j < model->m; j++){
			model->MCtheta[j][d] /= normsum;
			model->MCtheta[j][d] = 0.0;

			mcmc->logm[j][d] = log(model->alpha + mcmc->m[j][d]);
		}

	}

	//update beta, mu
	for (j = 0; j < model->m; j++){
		normsum = model->nu*model->n + mcmc->tbar[j];
		mcmc->logtbar[j] = log(normsum);
		for (n = 0; n < model->n; n++){
			model->MCbeta[j][n] = (model->nu + mcmc->t[j][n])/normsum;
			mcmc->logt[j][n] = log(model->nu + mcmc->t[j][n]);
		}

	}

  	free(theta);
  	free(beta);
  	free(z);
  	free(p);
  	free(alpha);
  	free(nu);

}


