python3 compare_datasets.py synthetic/aPC/PYR morphologies/aPC/PYR --min-variance-ratio 1.5 --min-relative-mean-diff 0.1
python3 compare_datasets.py synthetic/aPC/SL morphologies/aPC/SL --min-variance-ratio 1.5 --min-relative-mean-diff 0.1

python3 compare_datasets.py synthetic/OB/MITRAL morphologies/OB/MITRAL --min-variance-ratio 1.5 --min-relative-mean-diff 0.1
python3 compare_datasets.py synthetic/OB/TUFTED morphologies/OB/TUFTED --min-variance-ratio 1.5 --min-relative-mean-diff 0.1

python3 compare_datasets.py synthetic/Neocortex/PYR morphologies/Neocortex/PYR --min-variance-ratio 1.5 --min-relative-mean-diff 0.1 --merge-oblique-into-apical

python3 compare_datasets.py synthetic/Neocortex/PYR morphologies/Neocortex/PYR --min-variance-ratio 1.5 --min-relative-mean-diff 0.1
