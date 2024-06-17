# Lexical predictability analysis

Inspired by an approach proposed by Sharpe, Kuperberg et al. [1], this repository performs a lexical predictability analysis in which the predictability (or probability of prediction) of words in text samples are quantified and compared for varying amount of context available as well as psychotic levels (represented by textual incoherence).

Language abnormalities and schizophrenia are closely related, and incoherence is among the predictors often used in diagnosis [2]. Additionally, analysing psychotic speech with automated speech analysis or computational models allows for early prediction of psychotic events [2, 3]. Thus, lexical predictability, i.e. the probability that a word is predicted based on its prior context, could be a fruitful biomarker to assess incoherence.

Following [1], I analyze text samples word by word, and for lengths of prior context varying systematically from very local (one previous word) to very global (all available previous text), and I study the effect of disorder level (incoherence level) on the relation between context length and predictability.

Instead of patients data I simulate incoherent speech artificially by taking text samples from a free source and introduce increasing disorder levels, represented as randomly shuffling an increasing proportion of words with the text samples, and use the language model GPT-2 to assess words' predictability.

The mehtods and results are presented in [this report](https://github.com/clelf/lexical-predictability/blob/main/Linguistic_assignment_CLF.pdf). The code in the repository is organized as follows:

- ```generate_text_samples.py```: generates a .csv file containing 1000 samples (100 per disorder level)
- ```lexpred_multi.py```: computes the predictability scores for varying context length and disorder levels and save the scores as a .csv file
- ```results_analysis.py```: studies the relation of the different variables together and produce visualizations



**References**

[1] Sharpe, V., Ford, S., Nour-Eddine, S., Palaniyappan, L., & Kuperberg, G. (2023). Lexical predictability in schizophrenia: a computational approach to quantifying and
understanding thought disorder. *Abstract for Schizophrenia International Research Society*. [link](...)

[2] Hitczenko, K., Mittal, V. A., & Goldrick, M. (2021). Understanding  language abnormalities and associated clinical markers in psychosis: the promise of computational methods. *Schizophrenia Bulletin*, *47*(2), 344-362.

[3] Corcoran, C. M., Mittal, V. A., Bearden, C. E., Gur, R. E., Hitczenko,  K., Bilgrami, Z., ... & Wolff, P. (2020). Language as a biomarker  for psychosis: a natural language processing approach. *Schizophrenia research*, *226*, 158-166.



