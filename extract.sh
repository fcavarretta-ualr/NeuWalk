#python3 extract_apc.py morphologies/aPC/SL neuwalk/presets/anterior_piriform_cortex/semilunar.parameters.json 
#python3 extract_apc.py morphologies/aPC/PYR neuwalk/presets/anterior_piriform_cortex/pyramidal.parameters.json 
#python3 extract_ob.py morphologies/OB/MITRAL neuwalk/presets/olfactory_bulb/mitral.parameters.json 
#python3 extract_ob.py morphologies/OB/TUFTED neuwalk/presets/olfactory_bulb/middle_tufted.parameters.json 
#python3 extract_neo.py morphologies/Neocortex/PYR 

python3 calibrate_parameters.py neuwalk/presets/anterior_piriform_cortex/semilunar.parameters.json --sholl-threshold 0.5 --bifurcation-threshold 0.5 &
python3 calibrate_parameters.py neuwalk/presets/anterior_piriform_cortex/pyramidal.parameters.json --sholl-threshold 0.5 --bifurcation-threshold 0.5 &

python3 calibrate_parameters.py neuwalk/presets/olfactory_bulb/mitral.parameters.json --sholl-threshold 0.5 --bifurcation-threshold 0.5 &
python3 calibrate_parameters.py neuwalk/presets/olfactory_bulb/middle_tufted.parameters.json --sholl-threshold 0.5 --bifurcation-threshold 0.5 &

python3 calibrate_parameters.py neuwalk/presets/neocortex/pyramidal.parameters.json --sholl-threshold 0.5 --bifurcation-threshold 0.5 &
