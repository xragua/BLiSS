
require ("xspec");
xspec_xsect("vern");
xspec_abund("wilm");
load_xspec_local_models("/Users/graciela/Desktop/local_models/tbnew_2.3.2/absmodel/");
load_xspec_local_models("/opt/software/src/warmabstar/");
require("/Users/graciela/Desktop/local_models/isisscripts.sl");
require("/Users/graciela/Desktop/local_models/xstardb");
variable MODEL_DIR;
MODEL_DIR = "/Users/graciela/Desktop/git/laex_detect_lines_isis";
require(MODEL_DIR + "/create_line_model.sl");





load_data("med4.ds");
group_data(1,8);
xnotice_en(1,0.35,10);
xrange(0.35,10);
plot_data([1]);




fit_fun("tbnew(1)*powerlaw(1)");
fit_counts();

create_line_model("");
fit_fun("tbnew(1)*(powerlaw(1)+linemodel)");
fit_line_model(&eval_counts,0.005,30);
