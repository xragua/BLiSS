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

public define how_many_gaussian ()
{
    variable n = get_num_pars();
    variable i, info, k, matched, last = 0;
    for (i = 1; i <= n; i++)
      {
         info = get_par_info(i);
         matched = sscanf(info.name, "egauss(%d).%*s", &k);
         if (matched == 1 && k > last)
            last = k;
      }
    return last;
}

private define test_params (particle, evalfun)
{
    variable info, nbins, free_pars, chisquared;
    set_params(particle);
    () = @evalfun(&info);
    nbins = info.num_bins;
    free_pars = info.num_variable_params;
    chisquared = info.statistic / (nbins - free_pars);
    return chisquared;
}

public define create_line_model(name)
{
    write_plot("spec");

    variable cmd, model_file;
    cmd = "python -m bliss.isis_interface.run_bliss_for_isis " + name + " --spectrum spec_0.dat --output-dir .";
    system(cmd);

    model_file = "set_line_model_" + name + ".sl";
    () = evalfile(model_file);
    fit_fun("linemodel");
    vmessage("BLiSS line model loaded from %s", model_file);
}

public define fit_line_model(evalfun, threshold, max_lines, name)
{
    variable parameter_file = "set_line_parameters_" + name + ".sl";
    () = evalfile(parameter_file);

    variable num = how_many_gaussian();
    if (num > max_lines)
      {
        num = max_lines;
        vmessage("Limiting number of fitted line candidates to %i", num);
      }

    if (num <= 0)
      {
        vmessage("Invalid number of gaussians: %i", num);
        return;
      }

    variable best_p = get_params();
    variable best_stat = 1.0e99;
    variable i;
    vmessage("Number of line candidates detected: %i", num);

    for (i = 1; i < num + 1; i++)
      {
        vmessage("Iteration number %i of %i", i, num);
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

        vmessage("next stat: %f", new_stat);
        if (new_stat < stat - threshold)
          {
            vmessage("line accepted");
            best_stat = new_stat;
            best_p = new_p;
            set_params(new_p);
          }
        else
          {
            set_par(paramName_area, 0);
            freeze(paramName_area);
            freeze(paramName_sigma);
            freeze(paramName_center);
          }
      }

    set_params(best_p);

    variable cmd;
    save_par("raw_model.par");
    cmd = "python -m bliss.isis_interface.isis_model_cleaner raw_model.par clean_model.par";
    system(cmd);
    load_par("clean_model.par");
}
