# Drift SVM


## Bibliography
1. [Concept Drift description on Wikipedia](http://en.wikipedia.org/wiki/Concept_drift)
2. [A. Tsymbal, _The problem of concept drift: definitions and related work_](http://www.scss.tcd.ie/publications/tech-reports/reports.04/TCD-CS-2004-15.pdf) [**local**](bibliography/TCD-CS-2004-15.pdf)
3. [G. Forman, _Tackling Concept Drift by Temporal Inductive Transfer_](https://www.hpl.hp.com/techreports/2006/HPL-2006-20R1.pdf) [**local**](bibliography/HPL-2006-20R1.pdf)
4. [M. Masud, _Mining Concept-Drifting Data Stream to Detect Peer-to-Peer Botnet Traffic_](https://personal.utdallas.edu/~bxt043000/Publications/Technical-Reports/UTDCS-05-08.pdf) [**local**](bibliography/UTDCS-05-08.pdf)
5. [EDDM-IWKDDS-2006, M. Garcia, _Early Drift Detection Method in The International Workshop on Knowledge Discovery from Data Stream_, code and datasets](https://web.archive.org/web/20070322063617/http://iaia.lcc.uma.es/Members/mbaena/papers/eddm/)
6. [Weka, Single Classifier Drift](https://moa.cms.waikato.ac.nz/details/classification/using-weka/)
7. [J. Gama, _A Survey on Concept Drift Adaptation_](https://www.win.tue.nl/~mpechen/publications/pubs/Gama_ACMCS_AdaptationCD_accepted.pdf) [**local**](bibliography/Gama_ACMCS_AdaptationCD_accepted.pdf)
8. [R. Polikar,*Guest Editorial Learning in Nonstationary and Evolving Environments*](http://home.deib.polimi.it/alippi/pdf/guest_editorial_2014.pdf) [**local**](bibliography/guest_editorial_2014.pdf) [content](https://dblp.org/db/journals/tnn/tnn25)
9. [I. Zliobaite et al., *Active Learning with Drifting Streaming Data*](https://www.researchgate.net/publication/260354315_Active_Learning_With_Drifting_Streaming_Data) [**local**](bibliography/Active_Learning_With_Drifting_Streaming_Data.pdf)
10. [M. Wozniak et al., *Active Learning Classification With Drifted Streaming Data*](https://www.sciencedirect.com/science/article/pii/S187705091631002X) [**local**](bibliography/ActiveLearningClassificationWithDriftedStreamingData.pdf)
11. [S. Janardan et al., *Concept drift in Streaming Data Classification: Algorithms,
Platforms and Issues*](https://www.sciencedirect.com/science/article/pii/S1877050917326881), overview
12. [S. Zubin et al., *Concept Drift Detection and Adaptation with
Hierarchical Hypothesis Testing*](https://arxiv.org/pdf/1707.07821.pdf)

## Concept drift papers with code / benchmarks
1. [A. Saffari et al., *On-line Random Forests*](https://ieeexplore.ieee.org/document/5457447)
  * [source code that works, with data](https://github.com/amirsaffari/online-random-forests)
  * [presentation](https://pdfs.semanticscholar.org/7633/c95812dea716917e8b23b84df18e7b03614e.pdf)
  * [continuation, 2010, does not build](https://github.com/amirsaffari/online-multiclass-lpboost)
2. [A.P. Cassidy, *Calculating Feature Importance in Data Streams with Concept Drift using Online Random Forest*](http://www.ccri.com/wp-content/uploads/2014/10/final_with_watermark.pdf)
  * [got to it by this stackexchange topic](https://stats.stackexchange.com/questions/99744/benchmark-data-sets-for-concept-drift-where-important-predictors-independent-va)
3. [Datasets for concept drift](http://www.liaad.up.pt/kdus/products/datasets-for-concept-drift)
  * J. Gama's workgroup, explore around for papers

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
  - [x] define rectangular shift kernel and probe for a window of 20-40 patterns
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
