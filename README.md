# Target localization and tracking tools
Here, the algorithms used to compute the target position using Time Difference Of Arrival (TDOA) are explained

# src\TargetLocalizationSimulations
These files are used to simulate a target trajectory and compute its position based ond TDOA, we have used 4 base stations as landmarks but can be  modified. The algorithms used are:
1- Particle Filter (PF)
2- Maximum A Posterior (MAP) estimation
3- MAP with marginalizing old states (MAP-M)
4- Close-form Weighted Least Square (WLS)
5- Iterative method with Maximum Likelihood (ML) estimation
6- A combination between WLS and ML (WLS-ML)

#More information

I. Masmitja, J. Navarro, S. Gomariz, J. Aguzzi, B. Kieft, T. O’Reilly, K. Katija, P. J. Bouvet,
C. Fannjiang, M. Vigo, P. Puig, A. Alcocer, G. Vallicrosa, N. Palomeras, M. Carreras, J. del Rio,
J. B. Company, Mobile robotic platforms for the acoustic tracking of deep-sea demersal fishery
resources. Sci. Robot. 5, eabc3701 (2020).
