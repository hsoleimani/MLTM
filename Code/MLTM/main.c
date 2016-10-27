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


	seed = atoi(argv[1]);

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r, seed);

	printf("SEED = %ld\n", seed);

	MAXITER = 7000;
	BURNIN = 1000;
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
			if (strcmp(init, "load") == 0)
				strcpy(model_name, argv[8]);
			train(corpus_file, label_file, settings_file, init, dir, model_name);

			gsl_rng_free (r);
			return(0);
		}
		if (strcmp(task, "test")==0)
		{
			strcpy(settings_file,argv[4]);
			strcpy(model_name,argv[5]);
			strcpy(dir,argv[6]);
			test(corpus_file, settings_file, model_name, dir);
			gsl_rng_free (r);
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
	int ntopics, nclasses, ndocs;
	double doclkh, wrdlkh;
	double normsum, temp, psi_sigma, alpha_sigma, nu_sigma;
	int d, n, j, c, i;

	mltm_corpus* corpus;
	mltm_model *model = NULL;
	mltm_mcmc *mcmc = NULL;
	mltm_var *var = NULL;

	time_t t1,t2;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &ndocs);
	fscanf(fp, "N %d\n", &d);
	fscanf(fp, "T %d\n", &MAXITER);
	fscanf(fp, "burnin %d\n", &BURNIN);
	fscanf(fp, "alpha %lf\n", &temp);
	fscanf(fp, "nu %lf\n", &temp);
	fscanf(fp, "psi %lf %lf\n", &temp, &temp);
	fscanf(fp, "sent %d\n", &SENT);
	fscanf(fp, "P %d\n", &d);
	fscanf(fp, "psi_sigma %lf\n", &psi_sigma);
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
		mcmc->a = 1;
	}
	else if (strcmp(start, "load")==0){//not updated
		printf("load\n");

		load_model(model, corpus, mcmc, model_name); //not loading psi, alpha, ...
		mcmc->mcse_z = 1;
		mcmc->mcse_mu = 1;

	}
	int maxs = 0;
	int mins = 1e5;
	int s, maxLs = 0;
	for (d = 0; d < model->D; d++){
		if (corpus->docs[d].length > maxs)	maxs = corpus->docs[d].length;
		if (corpus->docs[d].length < mins)	mins = corpus->docs[d].length;
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
	var->xprime = malloc(sizeof(int)*maxLs);
	var->yprob = malloc(sizeof(double)*maxs); //max # sent
	var->numS = malloc(sizeof(double)*model->c);
	var->num_pick_sents = mins;
	var->pick_sents = malloc(sizeof(int)*mins);
	var->xvec = malloc(sizeof(int)*model->c);//malloc(sizeof(int*)*maxLs);
	var->numxvec = 0;//malloc(sizeof(int)*maxLs);
	var->phi_vec = (hit_times*) malloc(sizeof(hit_times)*maxs);
	/*for (i = 0; i < maxLs; i++){
		var->xvec[i] = malloc(sizeof(int)*model->c);
	}*/



	sprintf(filename, "%s/%03d", dir,(int)mcmc->n);
	printf("%s\n",filename);
	write_mltm_model(corpus, model, mcmc, filename);

	time(&t1);
	sprintf(filename, "%s/likelihood.dat", dir);
	lhood_fptr = fopen(filename, "w");

	int tt;
	double temp_accRate;
	double psi0prime, psi1prime, u, logP, logPprime, logH;
	double alphaprime, nuprime;
	double prev_alpha, prev_nu, prev_psi0, prev_psi1;
	//psi_sigma = 0.4;
	//alpha_sigma = 0.15;
	//nu_sigma = 0.5;

	wrdlkh = 0.0;
	document* doc;
	sentence* sent;
	do{

		mcmc->n += 1.0;
		if (mcmc->n > model->BurnIn){
			if ((int)(mcmc->n-model->BurnIn)%mcmc->b == 0){
				mcmc->zybar2 = 0.0;
				mcmc->zhat2 = 0.0;
				mcmc->znum = 0.0;
				mcmc->muhat2 = 0.0;
				mcmc->muybar2 = 0.0;
			}
		}

		for (d = 0; d < corpus->ndocs; d++){

			doc_mcmc_hmc(&(corpus->docs[d]), model, mcmc, d, var, 0);

		}


		//sample psi
		temp_accRate = 0;
		c = 0;
		logP = 0.0;
		logPprime = 0.0;
		psi0prime = exp((gsl_ran_gaussian(r, psi_sigma)) + log(model->psi[c][0]));
		psi1prime = exp((gsl_ran_gaussian(r, psi_sigma)) + log(model->psi[c][1]));
		for (c = 0; c < model->c; c++){
			logP += (model->psi0[0]-1)*(log(model->psi[c][0])+log(model->psi[c][1])) -
					(model->psi[c][0]+model->psi[c][1])/model->psi0[1] +
					model->m*(lgamma(model->psi[c][0]+model->psi[c][1])
							-lgamma(model->psi[c][0])-lgamma(model->psi[c][1]));
			logPprime += (model->psi0[0]-1)*(log(psi0prime)+log(psi1prime)) -
					(psi0prime+psi1prime)/model->psi0[1] +
					model->m*(lgamma(psi0prime+psi1prime)
							-lgamma(psi0prime)-lgamma(psi1prime));
			for (j = 0; j < model->m; j++){
				logP += lgamma(model->psi[c][0]+mcmc->K1[j][c])+lgamma(model->psi[c][1]+mcmc->K2[j][c])-
						lgamma(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]);
				logPprime += lgamma(psi0prime+mcmc->K1[j][c])+lgamma(psi1prime+mcmc->K2[j][c])-
						lgamma(psi0prime+mcmc->K1[j][c]+psi1prime+mcmc->K2[j][c]);
			}
		}
		logH = logPprime - logP;
		u = log(gsl_ran_flat(r, 0,1));
		for (c = 0; c < model->c; c++){
			if (u < min(0, logH)){
				temp_accRate = 1.0;
				model->psi[c][0] = psi0prime;
				model->psi[c][1] = psi1prime;
				for (j = 0; j < model->m; j++){
					mcmc->logmu[j][c] = log((model->psi[c][0]+mcmc->K1[j][c])/
							(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
					mcmc->log1mmu[j][c] = log((model->psi[c][1]+mcmc->K2[j][c])/
							(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
				}
			}
			if (mcmc->n > model->BurnIn){
				model->MCpsi[c][0] = ((mcmc->n-1.0-model->BurnIn)*model->MCpsi[c][0] +
						model->psi[c][0])/(mcmc->n-model->BurnIn);
				model->MCpsi[c][1] = ((mcmc->n-1.0-model->BurnIn)*model->MCpsi[c][1] +
						model->psi[c][1])/(mcmc->n-model->BurnIn);
			}
		}
		//temp_accRate /= model->c;
		mcmc->acceptrate_psi = ((mcmc->n-1)*mcmc->acceptrate_psi + temp_accRate)/mcmc->n;

		//sample alpha
		temp_accRate = 0;
		alphaprime = exp((gsl_ran_gaussian(r, alpha_sigma)) + log(model->alpha));
		logP = (model->psi0[0]-1)*log(model->alpha) - model->alpha/model->psi0[1] +
				model->D*lgamma(model->alpha*model->m)-model->D*model->m*lgamma(model->alpha);
		logPprime = (model->psi0[0]-1)*log(alphaprime) - alphaprime/model->psi0[1] +
				model->D*lgamma(alphaprime*model->m)-model->D*model->m*lgamma(alphaprime);
		for (d = 0; d < model->D; d++){
			temp = 0.0;
			for (j = 0; j < model->m; j++){
				logP += lgamma(model->alpha+mcmc->m[j][d]);
				logPprime += lgamma(alphaprime+mcmc->m[j][d]);
				temp += mcmc->m[j][d];
			}
			logP -= lgamma(model->alpha*model->m+temp);
			logPprime -= lgamma(alphaprime*model->m+temp);
		}
		logH = logPprime - logP;
		u = log(gsl_ran_flat(r, 0,1));
		if (u < min(0, logH)){
			temp_accRate += 1.0;
			model->alpha = alphaprime;
			for (d = 0; d < model->D; d++){
				for (j = 0; j < model->m; j++){
					mcmc->logm[j][d] = log(model->alpha + mcmc->m[j][d]);
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



		// update beta and mu estimates
		if (mcmc->n > model->BurnIn){
			tt = (int)(mcmc->n-model->BurnIn)%mcmc->b;

			// update posterior of beta and mu
			for (j = 0; j < model->m; j++){
				normsum = model->nu*model->n + mcmc->tbar[j];
				for (n = 0; n < model->n; n++){
					model->MCbeta[j][n] = ((mcmc->n-1.0-model->BurnIn)*model->MCbeta[j][n] +
							(model->nu+mcmc->t[j][n])/normsum)/(mcmc->n-model->BurnIn);
				}
				for (c = 0; c < model->c; c++){
					temp = (model->psi[c][0] + mcmc->K1[j][c])/
							(model->psi[c][0] + model->psi[c][1] + mcmc->K1[j][c] + mcmc->K2[j][c]);
					model->MCmu[j][c] = ((mcmc->n-1.0-model->BurnIn)*model->MCmu[j][c] + temp)/(mcmc->n-model->BurnIn);

					if (tt > 0)
						mcmc->YbarMCmu[j][c] = ((tt-1.0)*mcmc->YbarMCmu[j][c] + temp)/tt;
					else{
						mcmc->YbarMCmu[j][c] = ((mcmc->b-1.0)*mcmc->YbarMCmu[j][c] + temp)/mcmc->b;
						mcmc->muybar2 += pow(mcmc->YbarMCmu[j][c], 2.0);
						mcmc->muhat2 += pow(model->MCmu[j][c], 2.0);
					}

				}
			}


			if (tt == 0){
				mcmc->zybar2 /= mcmc->znum;
				mcmc->zhat2 /= mcmc->znum;
				mcmc->zmeanybar2 = ((mcmc->a-1.0)*mcmc->zmeanybar2 + mcmc->zybar2)/mcmc->a;
				mcmc->mcse_z = mcmc->b*(mcmc->zmeanybar2 - mcmc->zhat2);
				mcmc->mcse_z = sqrt(mcmc->mcse_z/(mcmc->n-model->BurnIn));
				//printf("%lf %lf, %lf\n", mcmc->zmeanybar2, mcmc->zhat2, mcmc->mcse_z);
				mcmc->muybar2 /= (model->c*model->m);
				mcmc->muhat2 /= (model->c*model->m);
				mcmc->mumeanybar2 = ((mcmc->a-1.0)*mcmc->mumeanybar2 + mcmc->muybar2)/mcmc->a;
				mcmc->mcse_mu = mcmc->b*(mcmc->mumeanybar2 - mcmc->muhat2);
				mcmc->mcse_mu = sqrt(mcmc->mcse_mu/(mcmc->n-model->BurnIn));
				//printf("%lf %lf, %lf\n", mcmc->mumeanybar2, mcmc->muhat2, mcmc->mcse_mu);
				mcmc->a += 1;
			}
		}

		if (((int)mcmc->n)%50 == 0){
			//compute likelihood
			wrdlkh = 0.0;
			doclkh = 0.0;
			for (d = 0; d < model->D; d++){
				doclkh = 0.0;
				doc = &(corpus->docs[d]);
				for (s = 0; s < doc->length; s++){
					sent = &(doc->sents[s]);
					for (i = 0; i < sent->length; i++){
						temp = 0.0;
						for (j = 0; j < model->m; j++)
							temp += model->MCtheta[j][d]*model->MCbeta[j][sent->words[i]];
						doclkh += log(temp);
					}
				}
				wrdlkh += doclkh;
			}
			sprintf(filename, "%s/%03d", dir,1);
			write_mltm_model(corpus, model, mcmc, filename);
			time(&t2);
			fprintf(lhood_fptr, "%d %e %5ld %lf %lf %lf %lf %lf %lf\n",
					(int)mcmc->n, wrdlkh, (int)t2-t1, mcmc->mcse_z, mcmc->mcse_mu, mcmc->acceptrate
					, mcmc->acceptrate_psi, mcmc->acceptrate_alpha, mcmc->acceptrate_nu);
			fflush(lhood_fptr);
			printf("***** MCMC ITERATION %d *****, MCStdEr_z = %lf, MCStdEr_mu = %lf\n",
					(int)mcmc->n, mcmc->mcse_z, mcmc->mcse_mu);
		}


		if ((mcmc->n > MAXITER) || ((mcmc->n > (model->BurnIn+1000)) && (mcmc->mcse_z < 0.02) && (mcmc->mcse_mu < 0.006)))
			break;

	}while(1);

	fclose(lhood_fptr);

	sprintf(filename, "%s/final", dir);

	write_mltm_model(corpus, model, mcmc, filename);

	sprintf(filename, "%s/final.b", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < model->D; d++){
		for (c = 0; c < model->c; c++){
			fprintf(fp, "%5.10lf ", corpus->docs[d].MCb[c]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

}

void sent_mcmc(sentence* sent, mltm_model* model, mltm_mcmc* mcmc, int s, int d, int tt,
		mltm_var* var, int test)
{

	int i, j, c, n, cc;
	int prevz, check;
	double maxval, normsum, u;

	for (i = 0; i < sent->length; i++){

		var->numxvec = 0;
		for (c = 0; c < model->c; c++){
			if (sent->x[c] == i){
				var->xvec[var->numxvec] = c;
				var->numxvec += 1;
			}
		}

		n = sent->words[i];

		//sample z
		prevz = sent->z[i];
		mcmc->m[prevz][d] -= 1.0;
		mcmc->logm[prevz][d] = log(model->alpha + mcmc->m[prevz][d]);
		if (test == 0){
			mcmc->t[prevz][n] -= 1.0;
			mcmc->tbar[prevz] -= 1.0;
			mcmc->logt[prevz][n] = log(model->nu + mcmc->t[prevz][n]);
			mcmc->logtbar[prevz] = log(model->nu*model->n + mcmc->tbar[prevz]);
			for (cc = 0; cc < var->numxvec; cc++){
				c = var->xvec[cc];
				mcmc->K1[prevz][c] -= sent->y[c];
				mcmc->K2[prevz][c] -= 1.0 - sent->y[c];
				mcmc->logmu[prevz][c] = log((model->psi[c][0]+mcmc->K1[prevz][c])/
						(model->psi[c][0]+mcmc->K1[prevz][c]+model->psi[c][1]+mcmc->K2[prevz][c]));
				mcmc->log1mmu[prevz][c] = log((model->psi[c][1]+mcmc->K2[prevz][c])/
						(model->psi[c][0]+mcmc->K1[prevz][c]+model->psi[c][1]+mcmc->K2[prevz][c]));
			}
		}
		maxval = -1e50;
		for (j = 0; j < model->m; j++){
			if (test == 0)
				var->phi[j] = mcmc->logt[j][n] - mcmc->logtbar[j] + mcmc->logm[j][d];
			else
				var->phi[j] = mcmc->logt[j][n] + mcmc->logm[j][d];

			for (cc = 0; cc < var->numxvec; cc++){
				c = var->xvec[cc];
				if (test == 0){
					if (sent->y[c] == 0)
						var->phi[j] += mcmc->log1mmu[j][c];
					else
						var->phi[j] += mcmc->logmu[j][c];
				}else{
					if (sent->y[c] == 0)
						var->phi[j] += mcmc->log1mmu[j][c];
					else
						var->phi[j] += mcmc->logmu[j][c];
				}
			}
			if (var->phi[j] > maxval)	maxval = var->phi[j];
		}
		normsum = 0.0;
		for (j = 0; j < model->m; j++){
			var->phi[j] = exp(var->phi[j] - maxval);
			normsum += var->phi[j];
		}
		u = gsl_ran_flat(r, 0, 1);
		check = 0;
		for (j = 0; j < model->m; j++){
			var->phi[j] /= normsum;
			if (j > 0)	var->phi[j] += var->phi[j-1];

			if (u <= var->phi[j]){
				check = 1;
				break;
			}
		}

		sent->z[i] = j;
		if (mcmc->n > model->BurnIn){
			sent->MCz[i] = ((mcmc->n-1.0-model->BurnIn)*sent->MCz[i] +
					(double)j)/(mcmc->n-model->BurnIn);

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
		}
		mcmc->m[j][d] += 1.0;
		mcmc->logm[j][d] = log(model->alpha + mcmc->m[j][d]);
		if (test == 0){
			mcmc->t[j][n] += 1.0;
			mcmc->tbar[j] += 1.0;
			mcmc->logt[j][n] = log(model->nu + mcmc->t[j][n]);
			mcmc->logtbar[j] = log(model->nu*model->n + mcmc->tbar[j]);
			for (cc = 0; cc < var->numxvec; cc++){
				c = var->xvec[cc];
				mcmc->K1[j][c] += sent->y[c];
				mcmc->K2[j][c] += 1.0 - sent->y[c];
				mcmc->logmu[j][c] = log((model->psi[c][0]+mcmc->K1[j][c])/
						(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
				mcmc->log1mmu[j][c] = log((model->psi[c][1]+mcmc->K2[j][c])/
						(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
			}

		}
		if (check == 0)	printf("logical error: doc = %d, sent = %d, i = %d, j = %d\n", d,s,i,j);
	}

	//sample x
	for (c = 0; c < model->c; c++){

		if (SENT == 0){
			i = 0;
		}
		else{
			for (j = 0; j < model->m; j++){ // to save time
				var->z[j] = 0;
				var->phi[j] = 0.0;
			}
			if (test == 0){
				i = sent->x[c];
				j = sent->z[i];
				mcmc->K1[j][c] -= sent->y[c];
				mcmc->K2[j][c] -= 1.0 - sent->y[c];
				mcmc->logmu[j][c] = log((model->psi[c][0]+mcmc->K1[j][c])/
						(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
				mcmc->log1mmu[j][c] = log((model->psi[c][1]+mcmc->K2[j][c])/
						(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
			}

			maxval = -1e50;
			for (i = 0; i < sent->length; i++){
				j = sent->z[i];
				if ((var->z[j] == 1)){ // save previously computed gammas
					var->gamma[i] =  var->phi[j];
					if (var->gamma[i] > maxval)	maxval = var->gamma[i];
					continue;
				}
				if (test == 0){
					if (sent->y[c] == 0)
						var->gamma[i] = mcmc->log1mmu[j][c];
					else
						var->gamma[i] = mcmc->logmu[j][c];
				}else{
					if (sent->y[c] == 0)
						var->gamma[i] = mcmc->log1mmu[j][c];//log(1.0 - model->MCmu[j][c]);
					else
						var->gamma[i] = mcmc->logmu[j][c];//log(model->MCmu[j][c]);
				}
				var->z[j] = 1;
				var->phi[j] = var->gamma[i];
				if (var->gamma[i] > maxval)	maxval = var->gamma[i];
			}
			normsum = 0.0;
			for (i = 0; i < sent->length; i++){
				var->gamma[i] = exp(var->gamma[i] - maxval);
				normsum += var->gamma[i];
			}
			/*var->gamma[0] /= normsum;
			for (i = 1; i < sent->length; i++){ //note starts at i = 1
				var->gamma[i] = var->gamma[i-1] + var->gamma[i]/normsum;
			}*/

			u = gsl_ran_flat(r, 0, 1);
			check = 0;
			for (i = 0; i < sent->length; i++){
				var->gamma[i] /= normsum;
				if (i > 0) var->gamma[i] += var->gamma[i-1];
				if (u <= var->gamma[i]){
					check = 1;
					break;
				}
			}
			if (check == 0) printf("Logical Error: doc = %d, sent = %d, class = %c, Ls = %d, phi[0] = %lf\n",
					d,s,c,sent->length, var->gamma[0]);
		}

		sent->x[c] = i;
		j = sent->z[i];
		if (mcmc->n > model->BurnIn)
			sent->MCx[c] = ((mcmc->n-1.0-model->BurnIn)*sent->MCx[c] +
					(double)i)/(mcmc->n-model->BurnIn);
		if ((test == 0) && (SENT == 1)){
			mcmc->K1[j][c] += sent->y[c];
			mcmc->K2[j][c] += 1.0 - sent->y[c];
			mcmc->logmu[j][c] = log((model->psi[c][0]+mcmc->K1[j][c])/
					(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
			mcmc->log1mmu[j][c] = log((model->psi[c][1]+mcmc->K2[j][c])/
					(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
		}

	}
}

int compare_function(const void *a,const void *b) {
	hit_times *x = (hit_times *) a;
	hit_times *y = (hit_times *) b;
	if (x->phi > y->phi)
		return 1;
	else if (x->phi < y->phi)
		return -1;
	else
		return 0;
}



void doc_mcmc_hmc(document* doc, mltm_model* model, mltm_mcmc* mcmc, int d,
		mltm_var* var, int test){

	int s, i, j, c, tt;
	int maxY, current;
	double phi, temp1, temp2, normsum;
	int numY, new_y, ss, P, p;
	double mt, logp_change, v2_new;


	tt = (int)(mcmc->n-model->BurnIn)%mcmc->b;

	for (s = 0; s < doc->length; s++){

		sent_mcmc(&(doc->sents[s]), model, mcmc, s, d, tt, var, test);

	}

	P = model->P;
	T_HMC = (P + 0.5)*M_PI;

	// sample Ysc (Block MH algorithm on all s)
	for (c = 0; c < model->c; c++){
		//sent = &(doc->sents[s]);

		if (test == 1){
			maxY = 0;
			for (s = 0; s < doc->length; s++){
				i = doc->sents[s].x[c];
				j = doc->sents[s].z[i];
				doc->sents[s].y[c] = gsl_ran_bernoulli(r, model->MCmu[j][c]);
				if (doc->sents[s].y[c] == 1) maxY = 1;
				if (mcmc->n > model->BurnIn)
					doc->sents[s].MCy[c] = ((mcmc->n-1.0-model->BurnIn)*doc->sents[s].MCy[c] +
						(double)doc->sents[s].y[c])/(mcmc->n-model->BurnIn);
			}
			if (mcmc->n > model->BurnIn)
				doc->MCb[c] = ((mcmc->n-1.0-model->BurnIn)*doc->MCb[c] +
					(double)maxY)/(mcmc->n-model->BurnIn);

			continue;
		}

		if (doc->b[c] == -1){
			maxY = 0;
			for (s = 0; s < doc->length; s++){
				i = doc->sents[s].x[c];
				j = doc->sents[s].z[i];
				current = doc->sents[s].y[c];
				mcmc->K1[j][c] -= current;
				mcmc->K2[j][c] -= 1 - current;
				phi = (model->psi[c][0] + mcmc->K1[j][c])/
						(model->psi[c][0] + mcmc->K1[j][c]+(model->psi[c][1] + mcmc->K2[j][c]));
				doc->sents[s].y[c] = gsl_ran_bernoulli(r, phi);

				current = doc->sents[s].y[c];
				mcmc->K1[j][c] += current;
				mcmc->K2[j][c] += 1 - current;

				mcmc->logmu[j][c] = log((model->psi[c][0] + mcmc->K1[j][c])/
						(model->psi[c][0] + mcmc->K1[j][c]+(model->psi[c][1] + mcmc->K2[j][c])));
				mcmc->log1mmu[j][c] = log((model->psi[c][1] + mcmc->K2[j][c])/
						(model->psi[c][0] + mcmc->K1[j][c]+(model->psi[c][1] + mcmc->K2[j][c])));

				if (doc->sents[s].y[c] == 1) maxY = 1;
				if (mcmc->n > model->BurnIn)
					doc->sents[s].MCy[c] = ((mcmc->n-1.0-model->BurnIn)*doc->sents[s].MCy[c] +
						(double)doc->sents[s].y[c])/(mcmc->n-model->BurnIn);
			}
			if (mcmc->n > model->BurnIn)
				doc->MCb[c] = ((mcmc->n-1.0-model->BurnIn)*doc->MCb[c] +
					(double)maxY)/(mcmc->n-model->BurnIn);

			continue;
		}

		if (doc->b[c] == 0){
			for (s = 0; s < doc->length; s++){
				if (mcmc->n > model->BurnIn)
					doc->sents[s].MCy[c] = ((mcmc->n-1.0-model->BurnIn)*doc->sents[s].MCy[c] +
						(double)doc->sents[s].y[c])/(mcmc->n-model->BurnIn);
			}
			doc->MCb[c] = 0.0;
			continue;
		}

		for (j = 0; j < model->m; j++){
			var->temp1[j] = 0.0;
			var->temp2[j] = 0.0;
		}
		numY = 0;
		for (s = 0; s < doc->length; s++){
			doc->v[s] = gsl_ran_gaussian(r,1);
			numY += doc->sents[s].y[c];

			j = doc->sents[s].z[doc->sents[s].x[c]];
			current = doc->sents[s].y[c];
			mcmc->K1[j][c] -= current;
			mcmc->K2[j][c] -= 1 - current;
			var->temp1[j] += current;
			var->temp2[j] += 1 - current;

			//compute initial hit times
			phi = atan2(doc->sents[s].HMCx[c], doc->v[s]);
			var->phi_vec[s].s = s;
			if (phi > 0)
				var->phi_vec[s].phi = M_PI - phi;
			else
				var->phi_vec[s].phi = -phi;
		}
		//sort hit times
		qsort (var->phi_vec, doc->length, sizeof(*var->phi_vec), compare_function);


		// the first T =pi time causes every coordinate to reach zero
		for (ss = 0; ss < doc->length; ss++){
			s = var->phi_vec[ss].s;
			mt = var->phi_vec[ss].phi;

			doc->v[s] =  doc->v[s]*cos(mt) - doc->sents[s].HMCx[c]*sin(mt);

			j = doc->sents[s].z[doc->sents[s].x[c]];
			current = doc->sents[s].y[c];
			temp1 = model->psi[c][0]+mcmc->K1[j][c] + var->temp1[j] - current;
			temp2 = model->psi[c][1]+mcmc->K2[j][c] + var->temp2[j] - (1-current);
			if (current == 0){
				logp_change = log(temp1) - log(temp2);
			}else{
				if (numY == 1) 	logp_change = -1e50;
				else logp_change = - log(temp1) + log(temp2);
			}

			v2_new = pow(doc->v[s],2.0) + 2*logp_change;

			if (v2_new > 0){
				doc->v[s] = sqrt(v2_new)*(1-2*current);
				new_y = 1 - current;
				numY += new_y - current;
				j = doc->sents[s].z[doc->sents[s].x[c]];
				var->temp1[j] += new_y - current;
				var->temp2[j] += current - new_y;
				doc->sents[s].y[c] = new_y;
			}else{
				doc->v[s] *= -1;
			}
		}


		// The next P-1 cycles of T=pi have known hit times and hit velocities
		for (p = 1; p < P; p++){
			for (ss = 0; ss < doc->length; ss++){
				s = var->phi_vec[ss].s;

				j = doc->sents[s].z[doc->sents[s].x[c]];
				current = doc->sents[s].y[c];
				temp1 = model->psi[c][0]+mcmc->K1[j][c] + var->temp1[j] - current;
				temp2 = model->psi[c][1]+mcmc->K2[j][c] + var->temp2[j] - (1-current);
				if (current == 0){
					logp_change = log(temp1) - log(temp2);
				}else{
					if (numY == 1) 	logp_change = -1e50;
					else logp_change = - log(temp1) + log(temp2);
				}

				v2_new = pow(doc->v[s],2.0) + 2*logp_change;

				if (v2_new > 0){
					doc->v[s] = sqrt(v2_new)*(1-2*current);
					new_y = 1 - current;
					numY += new_y - current;
					j = doc->sents[s].z[doc->sents[s].x[c]];
					var->temp1[j] += new_y - current;
					var->temp2[j] += current - new_y;
					doc->sents[s].y[c] = new_y;
				}
				// if the particle does not cross there are here two sign inversions for V[c].
				// the first one is from moving t = pi, the second for being reflected.
			}
		}

		// At this point, all the coordinates have moved a time hit_times[j].second + (P-1)*pi
		// and need to move 1.5*pi - hit_times[j].second.
		maxY = 0;
		for (ss = 0; ss < doc->length; ss++){
			s = var->phi_vec[ss].s;
			phi = var->phi_vec[ss].phi;

			if (phi < M_PI/2){

				j = doc->sents[s].z[doc->sents[s].x[c]];
				current = doc->sents[s].y[c];
				temp1 = model->psi[c][0]+mcmc->K1[j][c] + var->temp1[j] - current;
				temp2 = model->psi[c][1]+mcmc->K2[j][c] + var->temp2[j] - (1-current);
				if (current == 0){
					logp_change = log(temp1) - log(temp2);
				}else{
					if (numY == 1) 	logp_change = -1e50;
					else logp_change = - log(temp1) + log(temp2);
				}

				v2_new = pow(doc->v[s],2.0) + 2*logp_change;

				if (v2_new > 0){
					doc->v[s] = sqrt(v2_new)*(1-2*current);
					new_y = 1 - current;
					numY += new_y - current;
					j = doc->sents[s].z[doc->sents[s].x[c]];
					var->temp1[j] += new_y - current;
					var->temp2[j] += current - new_y;
					doc->sents[s].y[c] = new_y;
				}
				mt = M_PI/2 - phi;
			}
			else{
				mt= 3*M_PI/2 - phi;
			}

			doc->sents[s].HMCx[c] = doc->v[s]*sin(mt);

			if (doc->sents[s].HMCx[c] >= 0)
				doc->sents[s].y[c] = 1;
			else
				doc->sents[s].y[c] = 0;

			if (doc->sents[s].y[c] == 1)	maxY = 1;
			if (mcmc->n > model->BurnIn)
				doc->sents[s].MCy[c] = ((mcmc->n-1.0-model->BurnIn)*doc->sents[s].MCy[c] +
					(double)doc->sents[s].y[c])/(mcmc->n-model->BurnIn);

		}


		/*maxY = 0;
		for (s = 0; s < doc->length; s++){
			if (doc->sents[s].HMCx[c] >= 0)
				doc->sents[s].y[c] = 1;
			else
				doc->sents[s].y[c] = 0;


			if (doc->sents[s].y[c] == 1)	maxY = 1;
			if (mcmc->n > model->BurnIn)
				doc->sents[s].MCy[c] = ((mcmc->n-1.0-model->BurnIn)*doc->sents[s].MCy[c] +
					(double)doc->sents[s].y[c])/(mcmc->n-model->BurnIn);

		}*/

		for (j = 0; j < model->m; j++){
			mcmc->K1[j][c] += var->temp1[j];
			mcmc->K2[j][c] += var->temp2[j];

			mcmc->logmu[j][c] = log((model->psi[c][0]+mcmc->K1[j][c])/
					(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
			mcmc->log1mmu[j][c] = log((model->psi[c][1]+mcmc->K2[j][c])/
					(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
		}
		if (maxY != 1)
			printf("Y Constraint Error\n");
		doc->MCb[c] = 1.0;
	}

	//free(phi_vec);

	//update posterior estimate of theta
	if (mcmc->n > model->BurnIn){
		normsum = 0.0;
		for (j = 0; j < model->m; j++){
			normsum += model->alpha + mcmc->m[j][d];
		}
		for (j = 0; j < model->m; j++){
			model->MCtheta[j][d] = ((mcmc->n-1.0-model->BurnIn)*model->MCtheta[j][d] +
					(model->alpha+mcmc->m[j][d])/normsum)/(mcmc->n-model->BurnIn);
		}
	}


}


void test(char* dataset, char* settings_file, char* model_name, char* dir)
{

	FILE* lhood_fptr;
	FILE* fp;
	char string[100];
	char filename[100];
	char lblfile[100];
	int iteration;
	int d, c, ntopics, nclasses, ndocs, i, j;
	double doclkh, wrdlkh, temp;

	mltm_corpus* corpus;
	mltm_model *model = NULL;
	mltm_mcmc *mcmc = NULL;
	mltm_var* var = NULL;
	time_t t1,t2;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &ndocs);
	fscanf(fp, "N %d\n", &d);
	fscanf(fp, "T %d\n", &MAXITER);
	fscanf(fp, "burnin %d\n", &BURNIN);
	fscanf(fp, "alpha %lf\n", &temp);
	fscanf(fp, "nu %lf\n", &temp);
	fscanf(fp, "psi %lf %lf\n", &temp, &temp);
	fscanf(fp, "sent %d\n", &SENT);
	fclose(fp);


	corpus = read_data(dataset, nclasses, ndocs, 0, lblfile, ntopics);


	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);
	model = new_mltm_model(corpus, settings_file);
	mcmc = new_mltm_mcmc(model);

	corpus->nterms = model->n;

	//init
	test_initialize(corpus, model, mcmc, model_name);

	// set up the log likelihood log file
	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

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
	var->xprime = malloc(sizeof(int)*maxLs);
	var->yprob = malloc(sizeof(double)*maxs); //max # sent
	var->numS = malloc(sizeof(double)*model->c);
	var->num_pick_sents = 2; //doesn't matter here
	var->pick_sents = malloc(sizeof(int)*2);
	var->xvec = malloc(sizeof(int)*model->c);//malloc(sizeof(int*)*maxLs);
	var->numxvec = 0;//malloc(sizeof(int)*maxLs);
	var->phi_vec = (hit_times*) malloc(sizeof(hit_times)*maxs);

	iteration = 0;

	wrdlkh = 0.0;

	time(&t1);
	mcmc->a = 1;
	int tt;
	document* doc;
	sentence* sent;
	do{

		mcmc->n += 1.0;
		if (mcmc->n > model->BurnIn){
			if ((int)(mcmc->n-model->BurnIn)%mcmc->b == 0){
				mcmc->zybar2 = 0.0;
				mcmc->zhat2 = 0.0;
				mcmc->znum = 0.0;
				mcmc->muhat2 = 0.0;
				mcmc->muybar2 = 0.0;
			}
		}

		for (d = 0; d < model->D; d++){

			doc_mcmc_hmc(&(corpus->docs[d]), model, mcmc, d, var, 1);

		}
		if (mcmc->n > model->BurnIn){
			tt = (int)(mcmc->n-model->BurnIn)%mcmc->b;

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

		if (iteration%1000 == 0){
			wrdlkh = 0.0;
			//sprintf(filename, "%s/test%03d", dir,1);
			//write_mltm_model(corpus, model, mcmc, filename);
			sprintf(filename, "%s/test%03d.b", dir, 1);
			fp = fopen(filename, "w");
			for (d = 0; d < model->D; d++){
				doc = &(corpus->docs[d]);
				for (c = 0; c < model->c; c++){
					fprintf(fp, "%5.10lf ", doc->MCb[c]);
				}
				fprintf(fp, "\n");
				doclkh = 0.0;
				for (s = 0; s < doc->length; s++){
					sent = &(doc->sents[s]);
					for (i = 0; i < sent->length; i++){
						temp = 0.0;
						for (j = 0; j < model->m; j++)
							temp += model->MCtheta[j][d]*model->MCbeta[j][sent->words[i]];
						doclkh += log(temp);
					}
				}
				wrdlkh += doclkh;
			}
			fclose(fp);

			time(&t2);

			fprintf(lhood_fptr, "%d %e %5ld %lf\n",iteration, wrdlkh, (int)t2-t1, mcmc->mcse_z);
			fflush(lhood_fptr);
			printf("======= MCMC Iteration %d =======, MCStdEr_z = %lf\n",iteration, mcmc->mcse_z);

		}


		iteration ++;

		if ((iteration > MAXITER) || ((iteration > (model->BurnIn+1000)) && (mcmc->mcse_z < 0.02)))
			break;

	}while(1);
	//*************************************

	sprintf(filename, "%s/testfinal", dir);
	write_mltm_model(corpus, model, mcmc, filename);

	sprintf(filename, "%s/testfinal.b", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < model->D; d++){
		for (c = 0; c < model->c; c++){
			fprintf(fp, "%5.10lf ", corpus->docs[d].MCb[c]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

}


void test_initialize(mltm_corpus* corpus, mltm_model* model, mltm_mcmc* mcmc, char* model_name){

	int i, j, s, d, n, c;
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
			//update k
			/*for (s = 0; s < corpus->docs[d].length; s++){
				j = corpus->docs[d].sents[s].z[corpus->docs[d].sents[s].x[c]];
				mcmc->K1[j][c] += corpus->docs[d].sents[s].y[c];
				mcmc->K2[j][c] += 1.0 - corpus->docs[d].sents[s].y[c];
			}*/
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

}

mltm_model* new_mltm_model(mltm_corpus* corpus, char* settings_file)
{
	int n, j, c, d, ntopics, nclasses;
	double temp1, temp2;
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
	model->alpha = 0.0;
	model->nu = 0.0;
	model->psi0 = malloc(sizeof(double)*2);
	model->psi0[0] = 0.0;
	model->psi0[1] = 0.0;
	model->BurnIn = BURNIN;

	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &model->D);
	fscanf(fp, "N %d\n", &model->n);
	fscanf(fp, "T %d\n", &model->T);
	fscanf(fp, "burnin %d\n", &model->BurnIn);
	fscanf(fp, "alpha %lf\n", &model->alpha);
	fscanf(fp, "nu %lf\n", &model->nu);
	fscanf(fp, "psi %lf %lf\n", &temp1, &temp2);
	fscanf(fp, "sent %d\n", &d);
	fscanf(fp, "P %d\n", &model->P);
	model->psi0[0] = temp1;
	model->psi0[1] = temp2;
	fclose(fp);

	model->MCbeta = malloc(sizeof(double*)*model->m);
	model->MCtheta = malloc(sizeof(double*)*model->m);
	model->MCmu = malloc(sizeof(double*)*model->m);
	for (j = 0; j < model->m; j++){
		model->MCbeta[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			model->MCbeta[j][n] = 0.0;
		}
		model->MCtheta[j] = malloc(sizeof(double)*model->D);
		for (d = 0; d < model->D; d++){
			model->MCtheta[j][d] = 0.0;
		}

		model->MCmu[j] = malloc(sizeof(double)*model->c);
		for (c = 0; c < model->c; c++){
			model->MCmu[j][c] = 0.0;
		}
	}

	model->MCpsi = malloc(sizeof(double*)*model->c);
	model->psi = malloc(sizeof(double*)*model->c);
	for (c = 0; c < model->c; c++){
		model->MCpsi[c] = malloc(sizeof(double)*2);
		model->psi[c] = malloc(sizeof(double)*2);
		model->MCpsi[c][0] = 0.0;
		model->MCpsi[c][1] = 0.0;
		model->psi[c][0] = 0.0;
		model->psi[c][1] = 0.0;
	}

	return(model);
}

mltm_mcmc * new_mltm_mcmc(mltm_model* model){

	int n, j, c, d;

	mltm_mcmc* mcmc = malloc(sizeof(mltm_mcmc));

	mcmc->n = 0;
	mcmc->b = 100;
	mcmc->acceptrate = 0.0;
	mcmc->numYsampled = 0.0;

	mcmc->t = malloc(sizeof(double*)*model->m);
	mcmc->logt = malloc(sizeof(double*)*model->m);
	mcmc->tbar = malloc(sizeof(double)*model->m);
	mcmc->logtbar = malloc(sizeof(double)*model->m);
	mcmc->m = malloc(sizeof(double*)*model->m);
	mcmc->logm = malloc(sizeof(double*)*model->m);
	mcmc->K1 = malloc(sizeof(double*)*model->m);
	mcmc->K2 = malloc(sizeof(double*)*model->m);
	mcmc->logmu = malloc(sizeof(double*)*model->m);
	mcmc->log1mmu = malloc(sizeof(double*)*model->m);
	mcmc->YbarMCmu = malloc(sizeof(double*)*model->m);

	for (j = 0; j < model->m; j++){
		mcmc->tbar[j] = 0.0;
		mcmc->logtbar[j] = 0.0;
		mcmc->t[j] = malloc(sizeof(double)*model->n);
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

		mcmc->K1[j] = malloc(sizeof(double)*model->c);
		mcmc->K2[j] = malloc(sizeof(double)*model->c);
		mcmc->logmu[j] = malloc(sizeof(double)*model->c);
		mcmc->log1mmu[j] = malloc(sizeof(double)*model->c);
		mcmc->YbarMCmu[j] = malloc(sizeof(double)*model->c);
		for (c = 0; c < model->c; c++){
			mcmc->K1[j][c] = 0.0;
			mcmc->K2[j][c] = 0.0;
			mcmc->logmu[j][c] = 0.0;
			mcmc->log1mmu[j][c] = 0.0;
			mcmc->YbarMCmu[j][c] = 0.0;
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
		corpus->docs[nd].v = malloc(sizeof(double)*Sd);
		corpus->docs[nd].length = Sd;
		corpus->docs[nd].b = malloc(sizeof(int)*nclasses);
		corpus->docs[nd].MCb = malloc(sizeof(double)*nclasses);
		for (c = 0; c < nclasses; c++){
			corpus->docs[nd].b[c] = 0;
			corpus->docs[nd].MCb[c] = 0.0;
		}
		for (s = 0; s < Sd; s++){
			if (SENT == 1)
				fscanf(fileptr, "<%10d> ", &Ls);
			else
				Ls = 1;
			corpus->docs[nd].sents[s].words = malloc(sizeof(int)*Ls);
			corpus->docs[nd].sents[s].length = Ls;
			corpus->docs[nd].sents[s].z = malloc(sizeof(int)*Ls);
			corpus->docs[nd].sents[s].MCz = malloc(sizeof(double)*Ls);
			corpus->docs[nd].sents[s].YbarMCz = malloc(sizeof(double)*Ls);
			//corpus->docs[nd].sents[s].MCzvec = malloc(sizeof(double*)*Ls);
			corpus->docs[nd].sents[s].y = malloc(sizeof(int)*nclasses);
			corpus->docs[nd].sents[s].MCy = malloc(sizeof(double)*nclasses);
			corpus->docs[nd].sents[s].HMCx = malloc(sizeof(double)*nclasses); //For Hamiltonian MC
			corpus->docs[nd].sents[s].x = malloc(sizeof(int)*nclasses);
			corpus->docs[nd].sents[s].MCx = malloc(sizeof(double)*nclasses);
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
		printf("reading data from %s\n", lbl_filename);
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
	FILE* fptheta;
	FILE* fpz;
	FILE* fpx;
	FILE* fpy;
	FILE* fpMCz;
	FILE* fpMCx;
	int n, j, d, c, s, i;
	document* doc;
	sentence* sent;


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

	//psi
	sprintf(filename, "%s.psi", root);
	fileptr = fopen(filename, "w");
	for (c = 0; c < model->c; c++){
		fprintf(fileptr, "%5.10lf %5.10lf\n", model->MCpsi[c][0], model->MCpsi[c][1]);
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

	//mu
	sprintf(filename, "%s.mu", root);
	fileptr = fopen(filename, "w");
	for (j = 0; j < model->m; j++){
		for (c = 0; c < model->c; c++){
			fprintf(fileptr, "%5.10lf ", model->MCmu[j][c]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	//Y
	sprintf(filename, "%s.theta", root);
	fptheta = fopen(filename, "w");
	sprintf(filename, "%s.MCy", root);
	fileptr = fopen(filename, "w");
	sprintf(filename, "%s.z", root);
	fpz = fopen(filename, "w");
	sprintf(filename, "%s.y", root);
	fpy = fopen(filename, "w");
	sprintf(filename, "%s.x", root);
	fpx = fopen(filename, "w");
	sprintf(filename, "%s.MCz", root);
	fpMCz = fopen(filename, "w");
	sprintf(filename, "%s.MCx", root);
	fpMCx = fopen(filename, "w");

	for (d = 0; d < model->D; d++){

		for (j = 0; j < model->m; j++){
			fprintf(fptheta, "%.10lf ",model->MCtheta[j][d]);
		}
		fprintf(fptheta, "\n");

		doc = &(corpus->docs[d]);
		for (s = 0; s < doc->length; s++){

			sent = &(doc->sents[s]);

			for (c = 0; c < model->c; c++){
				if (c < model->c-1){
					fprintf(fpy, "%d ", sent->y[c]);
					fprintf(fpx, "%d ", sent->x[c]);
					fprintf(fpMCx, "%2.2lf ", sent->MCx[c]);
					fprintf(fileptr, "%2.2lf ",sent->MCy[c]);
				}
				else{
					fprintf(fileptr, "%2.2lf | ",sent->MCy[c]);
					fprintf(fpy, "%d |", sent->y[c]);
					fprintf(fpx, "%d |", sent->x[c]);
					fprintf(fpMCx, "%2.2lf |", sent->MCx[c]);
				}
			}

			for (i = 0; i < sent->length; i++){
				if (i < sent->length-1){
					fprintf(fpz, "%d ", sent->z[i]);
					fprintf(fpMCz, "%2.2lf ", sent->MCz[i]);
				}else{
					fprintf(fpz, "%d |", sent->z[i]);
					fprintf(fpMCz, "%2.2lf |", sent->MCz[i]);
				}
			}
   		}
		fprintf(fileptr, "\n");
		fprintf(fpx, "\n");
		fprintf(fpy, "\n");
		fprintf(fpz, "\n");
		fprintf(fpMCz, "\n");
		fprintf(fpMCx, "\n");
	}
	fclose(fileptr);
	fclose(fptheta);
	fclose(fpz);
	fclose(fpy);
	fclose(fpx);
	fclose(fpMCx);
	fclose(fpMCz);

	//state
	sprintf(filename, "%s.state", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "mcmc_n %d\n", (int)mcmc->n);
	fprintf(fileptr, "mcmc_a %d\n", (int)mcmc->a);
	fprintf(fileptr, "zmeanybar %lf\n", mcmc->zmeanybar2);
	fprintf(fileptr, "mumeanybar %lf\n", mcmc->mumeanybar2);
	fclose(fileptr);

}

void load_model(mltm_model* model, mltm_corpus* corpus, mltm_mcmc* mcmc, char* model_root){

	char filename[100];
	FILE* fileptr;
	FILE* fptheta;
	FILE* fpz;
	FILE* fpx;
	FILE* fpy;
	FILE* fpMCz;
	FILE* fpMCx;
	int j, n, c, d, i, s;
	//float x;
	double y;

	sprintf(filename, "%s.beta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			fscanf(fileptr, " %lf", &y);
			model->MCbeta[j][n] = y;
		}
	}
	fclose(fileptr);

	sprintf(filename, "%s.mu", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (j = 0; j < model->m; j++){
		for (c = 0; c < model->c; c++){
			fscanf(fileptr, "%lf ", &y);
			model->MCmu[j][c] = y;
		}
	}
	fclose(fileptr);

	sprintf(filename, "%s.theta", model_root);
	fptheta = fopen(filename, "r");
	sprintf(filename, "%s.MCy", model_root);
	fileptr = fopen(filename, "r");
	sprintf(filename, "%s.z", model_root);
	fpz = fopen(filename, "r");
	sprintf(filename, "%s.y", model_root);
	fpy = fopen(filename, "r");
	sprintf(filename, "%s.x", model_root);
	fpx = fopen(filename, "r");
	sprintf(filename, "%s.MCz", model_root);
	fpMCz = fopen(filename, "r");
	sprintf(filename, "%s.MCx", model_root);
	fpMCx = fopen(filename, "r");

	for (d = 0; d < model->D; d++){

		for (j = 0; j < model->m; j++){
			fscanf(fptheta, "%lf ", &model->MCtheta[j][d]);
		}

		for (s = 0; s < corpus->docs[d].length; s++){

			for (i = 0; i < corpus->docs[d].sents[s].length; i++){
				n = corpus->docs[d].sents[s].words[i];

				if (i < corpus->docs[d].sents[s].length-1){
					fscanf(fpz, "%d ", &corpus->docs[d].sents[s].z[i]);
					fscanf(fpMCz, "%lf ", &corpus->docs[d].sents[s].MCz[i]);
				}else{
					fscanf(fpz, "%d |", &corpus->docs[d].sents[s].z[i]);
					fscanf(fpMCz, "%lf |", &corpus->docs[d].sents[s].MCz[i]);
				}
				j = corpus->docs[d].sents[s].z[i];
				mcmc->t[j][n] += 1.0;
				mcmc->tbar[j] += 1.0;
				mcmc->m[j][d] += 1.0;
			}
			for (c = 0; c < model->c; c++){

				if (c < model->c-1){
					fscanf(fpy, "%d ", &corpus->docs[d].sents[s].y[c]);
					fscanf(fpx, "%d ", &corpus->docs[d].sents[s].x[c]);
					fscanf(fpMCx, "%lf ", &corpus->docs[d].sents[s].MCx[c]);
					fscanf(fileptr, "%lf ", &corpus->docs[d].sents[s].MCy[c]);
				}
				else{
					fscanf(fileptr, "%lf | ", &corpus->docs[d].sents[s].MCy[c]);
					fscanf(fpy, "%d |", &corpus->docs[d].sents[s].y[c]);
					fscanf(fpx, "%d |", &corpus->docs[d].sents[s].x[c]);
					fscanf(fpMCx, "%lf |", &corpus->docs[d].sents[s].MCx[c]);
				}

				j = corpus->docs[d].sents[s].z[corpus->docs[d].sents[s].x[c]];
				mcmc->K1[j][c] += corpus->docs[d].sents[s].y[c];
				mcmc->K2[j][c] += 1.0 - corpus->docs[d].sents[s].y[c];
			}
		}

		for (j = 0; j < model->m; j++){
			mcmc->logm[j][d] = log(model->alpha + mcmc->m[j][d]);
		}
	}
	for (j = 0; j < model->m; j++){
		mcmc->logtbar[j] = log(model->nu*model->n + mcmc->tbar[j]);
		for (n = 0; n < model->n; n++){
			mcmc->logt[j][n] = log(model->nu + mcmc->t[j][n]);
		}
		for (c = 0; c < model->c; c ++){
			mcmc->logmu[j][c] = log((model->psi[c][0]+mcmc->K1[j][c])/
					(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));
			mcmc->log1mmu[j][c] = log((model->psi[c][1]+mcmc->K2[j][c])/
					(model->psi[c][0]+mcmc->K1[j][c]+model->psi[c][1]+mcmc->K2[j][c]));

		}
	}
	fclose(fileptr);
	fclose(fptheta);
	fclose(fpz);
	fclose(fpy);
	fclose(fpx);
	fclose(fpMCx);
	fclose(fpMCz);

	sprintf(filename, "%s.state", model_root);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "mcmc_n %d\n", &n);
	mcmc->n = (double)n;
	fscanf(fileptr, "mcmc_a %d\n", &mcmc->a);
	fscanf(fileptr, "zmeanybar %lf\n", &mcmc->zmeanybar2);
	fscanf(fileptr, "mumeanybar %lf\n", &mcmc->mumeanybar2);
	fclose(fileptr);

	//return(model);
}


void random_initialize_model(mltm_corpus* corpus, mltm_model * model, mltm_mcmc* mcmc){

	int i, s, d, n, j, c;
	int maxY, check;
	double u, normsum;

	double* theta = malloc(sizeof(double)*model->m);
	double* p = malloc(sizeof(double)*model->m);
	double* alpha = malloc(sizeof(double)*model->m);
	int* z = malloc(sizeof(int)*model->m);
	double* nu = malloc(sizeof(double)*model->n);
	double* beta = malloc(sizeof(double)*model->n);


	//printf("doc %d, sent %d, word %d is %d\n", 5, 3, 2, corpus->docs[0].sents[0].words[0]);
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
			if (model->MCbeta[j][n] == 0) model->MCbeta[j][n] = EPS;
		}
	}
	for (c = 0; c < model->c; c++){
		model->psi[c][0] = 1;
		model->psi[c][1] = 1;
		for (j = 0; j < model->m; j++){
			model->MCmu[j][c] = gsl_ran_beta (r, model->psi0[0], model->psi0[1]);
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
			if (model->MCtheta[j][d] == 0)	model->MCtheta[j][d] = EPS;
		}
		//sample z
		for (s = 0; s < corpus->docs[d].length; s++){
			for (i = 0; i < corpus->docs[d].sents[s].length; i++){
				n = corpus->docs[d].sents[s].words[i];
				for (j = 0; j < model->m; j++){
					//p[j] = 1000.0*model->MCtheta[j][d]*model->MCbeta[j][n];
					p[j] = log(model->MCbeta[j][n]);
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
						mcmc->t[j][n] += 1.0;
						mcmc->tbar[j] += 1.0;
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
				do{
					//trial Ysc
					maxY = 0;
					for (s = 0; s < corpus->docs[d].length; s++){
						j = corpus->docs[d].sents[s].z[corpus->docs[d].sents[s].x[c]];
						u = gsl_ran_flat (r, 0, 1);
						if (u < model->MCmu[j][c]){
							corpus->docs[d].sents[s].y[c] = 1;
							do{
								corpus->docs[d].sents[s].HMCx[c] = gsl_ran_gaussian(r, 1);
								if (corpus->docs[d].sents[s].HMCx[c] > 0)
									break;
							}while(1);
							maxY = 1;
						}
						else{
							corpus->docs[d].sents[s].y[c] = 0;
							do{
								corpus->docs[d].sents[s].HMCx[c] = gsl_ran_gaussian(r, 1);
								if (corpus->docs[d].sents[s].HMCx[c] < 0)
									break;
							}while(1);
						}

					}
					//break if maxYsc = bsc
					if ((corpus->docs[d].b[c] == -1) || (maxY == corpus->docs[d].b[c]))
						break;

				}while(1);
			}
			//update k
			for (s = 0; s < corpus->docs[d].length; s++){
				j = corpus->docs[d].sents[s].z[corpus->docs[d].sents[s].x[c]];
				mcmc->K1[j][c] += corpus->docs[d].sents[s].y[c];
				mcmc->K2[j][c] += 1.0 - corpus->docs[d].sents[s].y[c];
			}
		}

		//update theta
		normsum = 0.0;
		for (j = 0; j < model->m; j++){
			mcmc->logm[j][d] = log(model->alpha + mcmc->m[j][d]);
			model->MCtheta[j][d] = model->alpha + mcmc->m[j][d];
			normsum += model->MCtheta[j][d];
		}
		for (j = 0; j < model->m; j++){
			model->MCtheta[j][d] /= normsum;
			model->MCtheta[j][d] = 0.0;
		}

	}

	//update beta, mu
	for (j = 0; j < model->m; j++){
		normsum = model->nu*model->n + mcmc->tbar[j];
		mcmc->logtbar[j] = log(model->nu*model->n + mcmc->tbar[j]);
		for (n = 0; n < model->n; n++){
			model->MCbeta[j][n] = (model->nu + mcmc->t[j][n])/normsum;
			mcmc->logt[j][n] = log(model->nu + mcmc->t[j][n]);
		}
		for (c = 0; c < model->c; c++){
			model->MCpsi[c][0] = model->psi[c][0];
			model->MCpsi[c][1] = model->psi[c][1];
			model->MCmu[j][c] = (model->psi[c][0] + mcmc->K1[j][c])/
					(model->psi[c][0]+model->psi[c][1]+mcmc->K1[j][c]+mcmc->K2[j][c]);
			mcmc->logmu[j][c] = log(model->MCmu[j][c]);
			mcmc->log1mmu[j][c] = log(1 - model->MCmu[j][c]);
		}
	}

	free(theta);
	free(beta);
	free(z);
	free(p);
	free(alpha);
	free(nu);

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

