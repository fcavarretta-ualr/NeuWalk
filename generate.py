import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from neuwalk.presets.olfactory_bulb.mitral import generate as generate_mitral
from neuwalk.presets.anterior_piriform_cortex.pyramidal import generate as generate_apc_pyramidal
from neuwalk.presets.olfactory_bulb.middle_tufted import generate as generate_tufted
from neuwalk.presets.anterior_piriform_cortex.semilunar import generate as generate_apc_semilunar
from neuwalk.presets.neocortex.pyramidal import generate as generate_neocortical_pyradamidal
import neuwalk.io.swc as swc


def _generate_one(func, filename_fmt, i):
    """Generate cell i and write it to disk, unless its file already exists. Runs in a worker thread."""
    filename = filename_fmt % i

    if os.path.exists(filename):
        return "skipped"

    result = func(i)
    swc.write_swc(filename, [result["output"]])
    return "generated"


def generate(func, filename_fmt, n=200, max_workers=None):
    """
    Generate n cells with func, writing each to its own SWC file.

    Cells are generated concurrently across a thread pool: every call to
    func (and the write_swc that follows it) runs in its own worker
    thread, so multiple cells are in flight at once rather than one at a
    time. Each cell is independent (its own seed, its own output file),
    so results don't depend on how the work happens to be interleaved.

    If cell i's output file already exists, it is skipped rather than
    regenerated, so an interrupted or extended run can be resumed by
    simply calling generate() again.

    Parameters
    ----------
    func : callable
        Preset generate function, called as func(seed).
    filename_fmt : str
        Format string with one %d placeholder for the cell index.
    n : int, default 200
        Number of cells to generate.
    max_workers : int, optional
        Number of worker threads. Defaults to os.cpu_count().
    """
    os.makedirs(os.path.dirname(filename_fmt) or ".", exist_ok=True)

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_one, func, filename_fmt, i): i
            for i in range(n)
        }

        for future in as_completed(futures):
            i = futures[future]

            try:
                status = future.result()
            except Exception as error:
                print('cell', i, 'FAILED:', error)
            else:
                print('cell', i, status)



#generate(generate_neocortical_pyradamidal, "synthetic/Neocortex/PYR.1.5SDs/cell%d.swc", max_workers=14)
#generate(generate_apc_semilunar, "synthetic/aPC/SL/cell%d.swc", max_workers=28)
#generate(generate_apc_pyramidal, "synthetic/aPC/PYR/cell%d.swc", max_workers=28)
#generate(generate_apc_semilunar, "synthetic/aPC/SL.1.5SDs/cell%d.swc", max_workers=14)
#generate(generate_apc_pyramidal, "synthetic/aPC/PYR.1.5SDs/cell%d.swc", max_workers=14)
#generate(generate_neocortical_pyradamidal, "synthetic/Neocortex/PYR/cell%d.swc", max_workers=14)
#generate(generate_apc_pyramidal, "synthetic/aPC/PYR/cell%d.swc", max_workers=14)
#generate(generate_mitral, "synthetic/OB/MITRAL/cell%d.swc")
generate(generate_tufted, "synthetic/OB/TUFTED/cell%d.swc")
