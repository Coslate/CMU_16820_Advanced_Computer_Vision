## HW2 - Lucas-Kanade Tracking

#### Instructions
For Q1.3 Please first go to code/ folder and use the following command line to reproduce the result stored in current folder
> python ./testCarSequence.py --threshold 1e-4
> python ./testCarSequence.py --threshold 1e-5
> python ./testGirlSequence.py --threshold 1e-4
> python ./testGirlSequence.py --threshold 1e-5

For Q1.4 Please first go to code/ folder and use the following command line to reproduce the result stored in current_folder
> python ./testCarSequenceWithTemplateCorrection.py --threshold 1e-4 --template_threshold 1
> python ./testCarSequenceWithTemplateCorrection.py --threshold 1e-4 --template_threshold 5
> python ./testCarSequenceWithTemplateCorrection.py --threshold 1e-4 --template_threshold 10
> python ./testCarSequenceWithTemplateCorrection.py --threshold 1e-5 --template_threshold 1
> python ./testCarSequenceWithTemplateCorrection.py --threshold 1e-5 --template_threshold 5
> python ./testCarSequenceWithTemplateCorrection.py --threshold 1e-5 --template_threshold 10

> python ./testGirlSequenceWithTemplateCorrection.py --threshold 1e-4 --template_threshold 1
> python ./testGirlSequenceWithTemplateCorrection.py --threshold 1e-4 --template_threshold 5
> python ./testGirlSequenceWithTemplateCorrection.py --threshold 1e-4 --template_threshold 10
> python ./testGirlSequenceWithTemplateCorrection.py --threshold 1e-5 --template_threshold 1
> python ./testGirlSequenceWithTemplateCorrection.py --threshold 1e-5 --template_threshold 5
> python ./testGirlSequenceWithTemplateCorrection.py --threshold 1e-5 --template_threshold 10

For Q2.3 Please first go to code/ folder and use the following command line to reproduce the result stored in the folder specified by --output_folder argument. (Please mkdir the output folder first.)
--use_inverse 0: use LucasKanadeAffine()
--use_inverse 1: use InverseCompositionAffine()

> python ./testAerialSequence.py --threshold 1e-11 --tolerance 0.064 --use_inverse 0 --output_folder ../result_aerialseq_th1e-11_tol0.064
> python ./testAerialSequence.py --threshold 1e-11 --tolerance 0.064 --use_inverse 1 --output_folder ../result_aerialseq_inverse_th1e-11_tol0.06
> python ./testAntSequence.py --threshold 1e-11 --tolerance 0.025 --use_inverse 0 --output_folder ../result_antseq_th1e-11_tol0.025
> python ./testAntSequence.py --threshold 1e-11 --tolerance 0.025 --use_inverse 1 --output_folder ../result_antseq_inverse_th1e-11_tol0.025