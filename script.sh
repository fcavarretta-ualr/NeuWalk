




python3 plot_neuron.py morphologies/OB/TUFTED/TT5.CNG.swc --ylim -1050 1050 --xlim -1050 1050 --zlim -1050 1050 --plot-2d yz --align-principal-axes --rotate-axis x --rotate-degree -90 --output ../../tufted-tt5-yz.png --hide-axes
python3 plot_neuron.py morphologies/OB/TUFTED/TT5.CNG.swc --ylim -1050 1050 --xlim -1050 1050 --zlim -1050 1050 --plot-2d xy --align-principal-axes --rotate-axis x --rotate-degree -90 --output ../../tufted-tt5-xy.png --hide-axes

python3 plot_neuron.py synthetic/OB/TUFTED/cell0.swc --ylim -1050 1050 --xlim -1050 1050 --zlim -1050 1050 --plot-2d xy --hide-axes --rotate-axis z --rotate-degree -180 --output ../../tufted-cell0-xy.png
python3 plot_neuron.py synthetic/OB/TUFTED/cell0.swc --ylim -1050 1050 --xlim -1050 1050 --zlim -1050 1050 --plot-2d xz --hide-axes --rotate-axis z --rotate-degree -180 --output ../../tufted-cell0-xz.png

python3 compare_sholl_plots.py morphologies/OB/TUFTED synthetic/OB/TUFTED --xlim -0 1600 --ylim 0 25 --font-size 9 --output ../../tufted-sholl-plot.png --exclude apical_dendrite --x-ticks 9 --y-ticks 6



python3 plot_neuron.py morphologies/OB/MITRAL/2M2.CNG.swc --ylim -1050 1050 --xlim -1050 1050 --zlim -1050 1050 --plot-2d yz --align-principal-axes --rotate-axis x --rotate-degree -90 --output ../../mitral-2m2-yz.png --hide-axes
python3 plot_neuron.py morphologies/OB/MITRAL/2M2.CNG.swc --ylim -1050 1050 --xlim -1050 1050 --zlim -1050 1050 --plot-2d xy --align-principal-axes --rotate-axis x --rotate-degree -90 --output ../../mitral-2m2-xy.png --hide-axes

python3 plot_neuron.py synthetic/OB/MITRAL/cell6.swc --ylim -1050 1050 --xlim -1050 1050 --zlim -1050 1050 --plot-2d yz --align-principal-axes --rotate-axis x --rotate-degree -90 --output ../../mitral-cell6-yz.png --hide-axes
python3 plot_neuron.py synthetic/OB/MITRAL/cell6.swc --ylim -1050 1050 --xlim -1050 1050 --zlim -1050 1050 --plot-2d xy --align-principal-axes --rotate-axis x --rotate-degree -90 --output ../../mitral-cell6-xy.png --hide-axes

python3 compare_sholl_plots.py morphologies/OB/MITRAL synthetic/OB/MITRAL --xlim -0 1600 --ylim 0 25 --font-size 9 --output ../../mitral-sholl-plot.png --exclude apical_dendrite --x-ticks 9 --y-ticks 6

python3 plot_neuron.py morphologies/aPC/PYR/d080414a-100.CNG.swc --labels apical_dendrite basal_dendrite --plot-2d xz --align-principal-axes --zlim -400 650 --xlim -525 525 --hide-axes --output ../../apcpyr-d080414a.png
python3 plot_neuron.py synthetic/aPC/PYR/cell2.swc --labels apical_dendrite basal_dendrite --plot-2d xz --align-principal-axes --zlim -400 650 --xlim -525 525 --hide-axes --output ../../apcpyr-cell2.png
python3 compare_sholl_plots.py morphologies/aPC/PYR synthetic/aPC/PYR --xlim -800 800 --ylim 0 30 --font-size 9 --x-ticks 9 --y-ticks 4 --output ../../apcpyr-sholl-plot.png --exclude-oblique


python3 plot_neuron.py morphologies/aPC/SL/d080421a-100.CNG.swc --labels apical_dendrite --plot-2d xz --align-principal-axes --rotate-degrees -5 --rotate-axis y --zlim 0 400 --xlim -200 200 --hide-axes --output ../../sl-d080421a.png
python3 plot_neuron.py synthetic/aPC/SL/cell0.swc --labels apical_dendrite --plot-2d xz --zlim 0 400 --xlim -200 200 --hide-axes --output ../../sl-cell0.png
python3 compare_sholl_plots.py morphologies/aPC/SL synthetic/aPC/SL --xlim -0 450 --ylim 0 20 --font-size 9 --output sl-sholl-plots.png --exclude basal_dendrite --x-ticks 10 --y-ticks 5 --output ../../sl-sholl-plot.png

python3 plot_neuron.py morphologies/Neocortex/PYR/C261296A-P1.CNG.swc --zlim -200 1000 --xlim -600 600 --plot-2d xz --align-principal-axes --output ../../neopyr-C261296A-P1-xz.png --hide-axes
python3 plot_neuron.py synthetic/Neocortex/PYR/cell46.swc --zlim -200 1000 --xlim -600 600 --plot-2d xz --align-principal-axes --output ../../neopyr-cell46-xz.png --hide-axes
python3 compare_sholl_plots.py morphologies/Neocortex/PYR synthetic/Neocortex/PYR --xlim -400 1200 --ylim 0 40 --font-size 9 --x-ticks 9 --y-ticks 5 --output ../../neopyr-sholl-plot.png


