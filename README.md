# Drift SVM


## Bibliography
1. [Concept Drift description on Wikipedia](http://en.wikipedia.org/wiki/Concept_drift)
2. [A. Tsymbal, _The problem of concept drift: definitions and related work_](http://www.scss.tcd.ie/publications/tech-reports/reports.04/TCD-CS-2004-15.pdf) [**local**](bibliography/TCD-CS-2004-15.pdf)
3. [G. Forman, _Tackling Concept Drift by Temporal Inductive Transfer_](https://www.hpl.hp.com/techreports/2006/HPL-2006-20R1.pdf) [**local**](bibliography/HPL-2006-20R1.pdf)
4. [M. Masud, _Mining Concept-Drifting Data Stream to Detect Peer-to-Peer Botnet Traffic_](https://personal.utdallas.edu/~bxt043000/Publications/Technical-Reports/UTDCS-05-08.pdf) [**local**](bibliography/UTDCS-05-08.pdf)
5. [EDDM-IWKDDS-2006, M. Garcia, _Early Drift Detection Method in The International Workshop on Knowledge Discovery from Data Stream_, code and datasets](https://web.archive.org/web/20070322063617/http://iaia.lcc.uma.es/Members/mbaena/papers/eddm/)
6. [Weka, Single Classifier Drift](https://moa.cms.waikato.ac.nz/details/classification/using-weka/)
7. [J. Gama, _A Survey on Concept Drift Adaptation_](https://www.win.tue.nl/~mpechen/publications/pubs/Gama_ACMCS_AdaptationCD_accepted.pdf) [**local**](bibliography/Gama_ACMCS_AdaptationCD_accepted.pdf)
8. [R. Polikar,*Guest Editorial Learning in Nonstationary and Evolving Environments*](http://home.deib.polimi.it/alippi/pdf/guest_editorial_2014.pdf) [**local**](bibliography/guest_editorial_2014.pdf) de adus tot

## Roadmap

### Code
- [x] classic IDSVM
  - [x] solution for two opposite patterns
  - [x] insert/remove a pattern from the solution
  - [x] compare g() for polynomial kernel 
  - [x] insert 500 USPS patterns (kernel)
  - [x] remove 480 USPS patterns (from now on, kernel only)
  - [x] adapt code to be device-independent (CUDA is faster)
  - [x] reduce C from 5.0 to 1.0 and observe error vectors, repeating learn/unlearn 500 patterns
- [ ] weighted IDSVM
  - [x] make C individual for each pattern, adapt Migration and extend its unit test
  - [x] also extend Migration unit test for lambda
  - [x] vary C for a specific pattern on basic 2-vector solution, probing for SVs, EVs and RVs
  - [ ] define rectangular shift kernel and probe for a window of 20-40 patterns
  - [ ] define a linear shift kernel (timing info is needed here, check bibliography)
  - [ ] define an exponential shift kernel (same as above)
  - [ ] comparison with SoA (TBD)

### Article
- [ ] add bibliography
- [ ] add latex template with todonotes package
- [ ] define main structure
- [ ] add theoretical considerations on incr/decr SVM
- [ ] refine for weighted SVM section
- [ ] TBD
