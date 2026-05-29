
%................................................................... PRIVATE
%.........................................................................

private define get_params ()
{
   variable i, n = get_num_pars ();
   variable p = Struct_Type [n];
   
   for (i = 1; i <= n; i++)
     {
    p[i-1] = get_par_info (i);

     }
   
   return p;
}

public define  how_many_gaussian ()
{
    variable n = get_num_pars();
    variable i, info, k, matched, last = 0;

    for (i = 1; i <= n; i++)
      {
         info = get_par_info(i);

         % Try to extract k from any egauss(k).<anything>
         matched = sscanf(info.name, "egauss(%d).%*s", &k);

         if (matched == 1 && k > last)
            last = k;
      }

    return last;   % 0 means: no egauss components found
}


private define test_params (particle, evalfun)
{
    variable info, nbins, free_pars, stat_value;
    
    set_params(particle);
    
    () = @evalfun(&info);
    
    nbins     = info.num_bins;
    free_pars = info.num_variable_params;
    
    variable l = get_fit_statistic();   % p.ej. "cash" o "chi2"

    
    if (l == "cash")
    {

        stat_value = info.statistic;
    }
    else
    {

        if (nbins > free_pars)
            stat_value = info.statistic / (nbins - free_pars);
        else
            stat_value = info.statistic;  % fallback por si acaso
    }
    
    return stat_value;
}

%...................................................................PUBLIC
%.........................................................................
public define create_line_model(name) {
    write_plot("spec");
    variable cmd;
    cmd = "python " + MODEL_DIR + "/lines.py " + name;
    system(cmd);

    cmd = "python " + MODEL_DIR + "/lines.py";
    system(cmd);
    
    () = evalfile("set_line_model_.sl");
    vmessage("Lines saved in model under name linemodel");
}



    
public define fit_line_model(evalfun,threshold,max_lines) {

    () = evalfile("set_line_parameters_.sl");

    variable num = how_many_gaussian();
    
    if (num>max_lines){
        num=max_lines;
        vmessage("Trying to fit");
    }

    
    if (num <= 0) {
       vmessage("Invalid number of gaussians: %i", num);
       return;
    }
    
    % Preallocate array to store stat - threshold for each iteration
    variable stat_diff = Double_Type[num];
    
    variable best_p = get_params();
    variable best_stat = test_params(best_p, evalfun);

    variable i;
    vmessage("number of line candidates detected: %i", num);
    
    
    for (i = 1; i < num+1; i++) {
        vmessage("iteration number %f of %f",i,num);
    
        variable p = get_params();
        variable stat = test_params(p, evalfun);

        variable paramName_area = sprintf("egauss(%i).area", i);
        variable paramName_sigma = sprintf("egauss(%i).sigma", i);
        variable paramName_center = sprintf("egauss(%i).center", i);
        
        thaw(paramName_area);
        thaw(paramName_sigma);
        thaw(paramName_center);
        
        fit_counts();
        variable new_p = get_params();
        variable new_stat = test_params(new_p, evalfun);
        
        % store this iteration's value:
        stat_diff[i-1] = new_stat - best_stat;   % or stat - threshold
        
        vmessage("next stat: %f", best_stat);
        
        
        if (new_stat < stat-threshold) {
            vmessage("line accepted");

            best_stat = new_stat;
            best_p = new_p;
            set_params(new_p);
            }
        else{
            set_par(paramName_area, 0);
            freeze(paramName_area);
            freeze(paramName_sigma);
            freeze(paramName_center);
            }
    }
  % Save staff diff for each line----------------
  variable fp = fopen("stat_diff.txt", "w");
  variable t;

  for (t = 0; t < length(stat_diff); t++) {
    fprintf(fp, "%g\n", stat_diff[t]);
  }

  fclose(fp);
  %----------------------------------------------
  
  set_params(best_p);
  
  variable cmd;
  save_par("raw_model.par");
  cmd = "python " + MODEL_DIR + "/clean_egauss.py raw_model.par clean_model.par";
  system(cmd);
  load_par("clean_model.par");

}
