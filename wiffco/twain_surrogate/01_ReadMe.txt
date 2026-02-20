

Load Surrogate Model
Adrien - TUM 

Inputs are [SAWS_up SAWS_right SAWS_down SAWS_left SATI_up SATI_right SATI_down SATI_left Yaw PowerDemand]

Sector-Averaged Wind Speeds (SAWS) in m/s
Sector-Averaged Turbulence Intensities (SATI) in %
Yaw in degrees
PowerDemand in % of rated power (100% is 3.35MW, 80% is 2.68MW, etc.) 


How 2 run

Python version
(first run the script of the desired ANN function, to declare the functions), then
Inputs = np.array([6.2563, 5.9030, 5.9930, 6.4745, 16.5359, 17.5766, 17.1182, 16.7116, 0, 100.0000])
DEL_BladeRoot_prediction = ANN_DEL_BladeRoot(Inputs)
DEL_TowerBase_prediction = ANN_DEL_TowerBase(Inputs)
Etc. 

The functions are already vectorized, i.e. Inputs can have more than one line for different entries, it will return the same number of outputs. 

