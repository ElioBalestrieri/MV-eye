# MV-eye
repo for MultiVariate EYEs closed/open

### basic repos structure
names perhaps self explanatory

**input/output folders**
serving for storage, but nor for data. likely absent in the actual github repo due to gitignore.

- STRG_ * 
		data 
		computed_features
		decoding_accuracy
       
**stable code**
the tested code to be used with parallel computing (preferably on HPC)

- STABLE_multithread

**supporting functions**
everything that gets called from the main codes

- helper_functions
  _it contains the cfg file !!!_
- plotting_functions

**sandboxes**
to try things out, either for feature generation or decoding. 

- sandbox_ *
		decode
		features
		
### branching regulations

HPC vs main (vs specific machines)

