
## BLiSS for ISIS

### Installation
---------------------------------------------------------

We created a tool for ISIS that allows to authomatically add emisison lines in the spectra. This module requires **isissscripts**.


### Workflow

1. **`create_line_model(name)`** generates and saves an initial line model using Python scripts.

2. **`fit_line_model(evalfun, threshold, max_lines)`** iterates through potential line candidates, adjusting parameters (such as area, sigma, and center of Gaussians), and selects the best model based on the reduced chi-square statistic acepting as line candidates those that al least improve the reduced chi-square by the threshold proposed. The lines will be tested by decreasing amplitud (line candidates with higher amplitude will be tested earlier) and the maximun number of lines to test should be provided by **max_lines**. The number of lines to be tested would be the minimun between the number of line candidates detected and the **max_lines** provided. Emission lines will be added to the continum and added to the model

The functions make use of system commands, Python scripts, and chi-square evaluations to fit the best Gaussian line model to the data.

### Use example with ISIS:

1. Download and save scripts in https://github.com/xragua/bliss/tree/main/bliss_for_isis in MODEL_DIR. Enter in ISIS and load the required scripts.

    ```isis
    isis> require("isisscripts.sl")
    isis> variable MODEL_DIR;
    isis> MODEL_DIR = "/our_local_models_path/detect_lines_isis/";
    isis> require(MODEL_DIR + "/create_line_model.sl");
    ```

2. As we would normally do, we load our spectra and make a plot, rebinning it to our preference:

    ```isis
    isis> load_data("spec.ds");
    isis> group(1;sn=12);
    isis> plot_data(1);
    ```

3. We can now create a model containing our emission lines, named **linemodel**, ready to use with other components as required by our spectra. Apart from this model, if we provide a name, we will save this line model in a file named `set_line_model_+name+.sl` in our folder to recover the complete model in subsequent runs:

    ```isis
    isis> create_line_model("test");
    ```

4. We define our complete model, as an example, that could be:

    ```isis
    isis> fit_fun("(tbnew(1)+constant(1)*tbnew(2))*(powerlaw(1)+bbody(1)+bbody(2)+linemodel)");
    ```

5. We fit our lines. Several line candidates are suggested, but only those which improve the fit by an absolute value of χ² = 0.005 are kept as free parameters, with an area different than 0:

    ```isis
    isis> fit_line_model(&eval_counts,0.005,10);
    ```

6. Once we are satisfied, we save our model as usual:

    ```isis
    isis> save_par("example_model.par");
    ```

### Recover our saved model example:

1. We recover the complete model as ususal:

    ```isis
    isis> load_par("example_model.par");
    ``
