import numpy as np


def _pad(values, size):
    padded_values = np.zeros(size)
    padded_values[:values.size] = values
    return padded_values


def extract_statistics(morphologies, bin_size):
    """Extract event, Sholl, and length statistics from processed morphologies."""
    morphologies = list(morphologies)

    if not morphologies:
        raise ValueError("At least one morphology is required.")

    records = []
    primary_counts = []
    total_lengths = []

    for roots in morphologies:
        sholl = []
        bifurcations = []
        annihilations = []
        internal_bifurcations = []
        
        for root in roots:
            root_sholl = root.sholl_plot(bin_size)
            bif, ann, internal = root._event_counts(bin_size)
            sholl.append(root_sholl)
            bifurcations.append(bif)
            annihilations.append(ann)
            internal_bifurcations.append(internal)

        sholl_size = max(map(len, sholl), default=1)
        event_size = sholl_size
        total_sholl = sum((_pad(values, sholl_size) for values in sholl), start=np.zeros(sholl_size))
        total_bif = sum((_pad(values, event_size) for values in bifurcations), start=np.zeros(event_size))
        total_ann = sum((_pad(values, event_size) for values in annihilations), start=np.zeros(event_size))
        total_internal = sum((_pad(values, event_size) for values in internal_bifurcations), start=np.zeros(event_size))

        records.append((total_sholl, total_bif, total_ann, total_internal))
        primary_counts.append(int(total_sholl[0]))
        total_lengths.append(sum(root.total_length for root in roots))

    sholl_size = max(len(record[0]) for record in records)
    event_size = sholl_size
    sholl_matrix = np.vstack([_pad(record[0], sholl_size) for record in records])
    bif_matrix = np.vstack([_pad(record[1], event_size) for record in records])
    ann_matrix = np.vstack([_pad(record[2], event_size) for record in records])
    internal_matrix = np.vstack([_pad(record[3], event_size) for record in records])

    mean_sholl = sholl_matrix.mean(axis=0)
    exposure = mean_sholl[:-1] * bin_size
    internal_density = internal_matrix.mean(axis=0) / bin_size
    no_bifurcation = np.isclose(bif_matrix.sum(axis=0), 0)
    no_annihilation = np.isclose(ann_matrix.sum(axis=0), 0)
    bifurcation_counts = bif_matrix.sum(axis=1)
    total_lengths = np.asarray(total_lengths, dtype=float)

    return {
        "sholl_plot": {"mean": mean_sholl, "std": sholl_matrix.std(axis=0)},
        "bifurcation_count": {"mean": float(bifurcation_counts.mean()), "std": float(bifurcation_counts.std())},
        "total_length": {"mean": float(total_lengths.mean()), "std": float(total_lengths.std())},
        "primary_count_range": {"min": int(np.min(primary_counts)), "max": int(np.max(primary_counts))},
        "bifurcation_internal_density": internal_density,
        "no_bifurcation_bins": no_bifurcation.tolist(),
        "no_annihilation_bins": no_annihilation.tolist(),
    }
