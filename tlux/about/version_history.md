|Version and Date       | Description           |
|-----------------------|-----------------------|
| 0.0.0<br>February 2022 | Initial commit. |
| 0.0.1<br>February 2022 | Added plotting library from util.plot and updated <br> requirements. |
| 0.0.2<br>February 2022 | Fixing pip install errors. |
| 0.0.2<br>February 2022 | Fixing pip install errors. |
| 0.0.2<br>February 2022 | Fixing pip install errors. |
| 0.0.2<br>February 2022 | Fixing pip install errors. |
| 0.0.3<br>February 2022 | Added well spaced random designs over the box and <br> ball. |
| 0.0.4<br>February 2022 | Added 'approximate' and support for generic surface <br> points in 'Plot.add_function'. |
| 0.0.5<br>March 2022 | APOS model added to library. |
| 0.0.6<br>March 2022 | Added math library utilities, included update ratio <br> in APOS model training record. |
| 0.0.7<br>March 2022 | Added missing import to random box function. |
| 0.0.8<br>March 2022 | Updating setup script to attempt Fortran compilation <br> early. |
| 0.0.9<br>March 2022 | Updated memory layout and incorporated SVD into data <br> normalization. Modified adaptive parameter update to <br> use a sliding linear number of parameters instead of <br> the binary search protocol. |
| 0.0.9<br>March 2022 | Updated memory layout and incorporated SVD into data <br> normalization. Modified adaptive parameter update to <br> use a sliding linear number of parameters instead of <br> the binary search protocol. |
| 0.0.10<br>March 2022 | Removing APOS compiler overrides. |
| 0.0.11<br>March 2022 | Made initial apositional model output normalized <br> correctly to zero mean and unit variance. |
| 0.0.12<br>April 2022 | Updated work size to int64 type. |
| 0.0.13<br>April 2022 | Preventing negative array sizes and numbers of <br> threads. |
| 0.0.14<br>April 2022 | Cleaned APOS memory usage, disabled compiled fmath, <br> added testing codes for least squares fits. |
| 0.0.15<br>April 2022 | Added large format testing for APOS and parallelized <br> the evaluation of internal state rank. |
| 0.0.16<br>April 2022 | Modified APOS fit routines to allow weighting the <br> value of individual Y with the new parameter YW. |
| 0.0.17<br>April 2022 | Minor patch, prevent division by zero in <br> CONDITION_MODEL. |
| 0.0.18<br>April 2022 | Implemented numpy evaluation class without need for <br> compiled code. Added EQUALIZE_Y configuration to <br> determine whether Y principal components are <br> flattened. Made default <br> ORTHOGONALIZING_STEP_FREQUENCY=0 to disable the <br> orthogonalizing entirely. |
| 0.0.19<br>April 2022 | Fixed a few bugs introduced after version 15. Now <br> refactored code still converges correctly for <br> apositional inputs. |
| 0.0.20<br>May 2022 | Added fast ball tree nearest neighbor codes to the <br> 'approximate' subpackage. |
| 0.0.21<br>May 2022 | Cleaning repository and pushing some waiting APOS <br> safeguards. Next step is well founded basis <br> replacement. |
| 0.0.22<br>June 2022 | Renamed APOS to AXY. Fixed error in aggregator <br> output mean calculation. Refactored CONDITION_MODEL <br> subroutines in preparation for further development. |
| 0.0.23<br>June 2022 | Added Data module. |
| 0.0.24<br>June 2022 | Added edge case checks for empty data, updated <br> memory usage patterns to improve model throughput <br> for large amounts of data, temporarily disabled all <br> local allocations that relate to in-development <br> code. Began implementing random pair generation for <br> pairwise batched aggregation. |
| 0.0.25<br>July 2022 | Cleaning up build code for package, relocating to <br> setup file. Added Timer class from util.system into <br> this package. Removed redundant variables from AXY <br> model for handling categoricals. |
| 0.0.26<br>July 2022 | Patching AXY build and import bugs. |
