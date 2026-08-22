import inspect
import numpy as np

from ... import misc
from ...random import Random
from ...core.topology import SectionSynthesizer as TopologySectionSynthesizer
from ...core.section import Section
from ...core.morphology import SectionSynthesizer as MorphologySectionSynthesizer


class MorphologySynthesizer:
    """
    Generate MorphologySectionSynthesizer trajectories from a
    TopologySectionSynthesizer tree.

    ``theta``, ``phi``, ``axis_direction``, ``elongation_bias``,
    ``bifurcation_bias``, ``bifurcation_internal_bias``, ``centrifugal``,
    ``max_angle``, ``elongation_random_weight``,
    ``elongation_bias_weight``, and ``correction_type`` each accept
    either a single value (applied regardless of label) or a dict
    mapping label to value, for example::

        elongation_bias={
            "apical_dendrite": apic_elongation_bias,
            "basal_dendrite": basal_elongation_bias,
            "apical_oblique": obl_elongation_bias,
        }

    Each section resolves its own value from these dicts using its own
    label, falling back to a ``"default"`` entry if present, otherwise
    raising ``KeyError``. All resolution happens inside this class: a
    dict is never passed to ``MorphologySectionSynthesizer`` itself,
    which only ever receives a single, already-resolved value for every
    one of these parameters.

    ``elongation_bias``, ``bifurcation_bias``, ``bifurcation_internal_bias``,
    ``centrifugal``, ``max_angle``, ``elongation_random_weight``,
    ``elongation_bias_weight``, ``axis_direction``, and
    ``correction_type`` are resolved twice: once when each primary
    section's ``MorphologySectionSynthesizer`` is constructed, and again
    for any section created by a bifurcation, using that section's own
    (possibly different) label, right after it is assigned. This is why
    a section created by a bifurcation never silently keeps a value
    meant for its parent's label.

    ``theta`` and ``phi`` are different: they control how primary-section
    directions are generated jointly, before any individual section
    exists, so they are resolved once, before any section is
    constructed. Primary sections are grouped by label, and each group's
    directions are generated together (evenly distributed over that
    group's own ``theta``/``phi`` range, then rotated by that group's
    own ``axis_direction``), independently from every other label's
    group. This replaces the previous count-keyed convention for
    ``theta``/``phi`` (a dict used to be keyed by the *number* of
    primary sections; it is now keyed by *label*, like every other
    resolvable parameter here). ``axis_direction`` is resolved for this
    purpose too (once per label group), in addition to being resolved
    per section as described above.
    """
    def _merge_profiles(self, profile_roots):

        assert sum(r.label != "soma" for r in profile_roots) == len(profile_roots), "Soma should be provided alone rather than inside a list"
        
        # for non oblique, roots are attached to the soma          
        profile_soma = TopologySectionSynthesizer(1, label="soma")
        for root in profile_roots:
            root.connect(profile_soma, relation="parent")
            
        return profile_soma

    @staticmethod
    def _resolve(value, label):
        """Return ``value`` as-is, or ``value[label]``/``value["default"]`` if it is a dict."""
        if not isinstance(value, dict):
            return value

        if label in value:
            return value[label]

        if "default" in value:
            return value["default"]

        raise KeyError(
            f"No entry for label {label!r} (and no 'default' entry) in {value!r}."
        )

    def __init__(
        self,
        topology,
        rng,
        origin=None,
        theta=(0.0, np.pi),
        phi=(0.0, 2.0 * np.pi),
        axis_direction=None,
        elongation_bias=None,
        bifurcation_bias=None,
        bifurcation_internal_bias=None,
        centrifugal=True,
        max_angle=np.pi / 2,
        elongation_random_weight=1.0,
        elongation_bias_weight=1.0,
        correction_type=None,
        parent=None,
    ):
        """
        Initialize the morphology synthesizer.

        Parameters
        ----------
        topology : TopologySectionSynthesizer
            Root of the synthesized topology. It may be either a soma or a
            single primary section.
        rng : numpy.random.Generator-like
            Random number generator used by the MorphologySectionSynthesizer objects.
        origin : array_like, optional
            Soma position. Default is ``[0, 0, 0]``.
        theta : float, sequence of two floats, or dict, optional
            Polar angle or angular range used to generate the primary
            directions. May also be a dict mapping label to any of the
            above, resolved per group of same-labeled primary sections
            (see the class docstring).
        phi : float, sequence of two floats, or dict, optional
            Azimuthal angle or angular range used to generate the primary
            directions. May also be a dict mapping label to any of the
            above, resolved the same way as ``theta``.
        axis_direction : array_like or dict, optional
            Direction of the local axial frame in global coordinates.
            May also be a dict mapping label to direction. Also used, per
            section, as the reference direction for a
            ``correction_type="somatodendritic"`` section (see below).
        elongation_bias : ElongationBias, sequence, or dict, optional
            Elongation bias or weighted elongation biases. May also be a
            dict mapping label to any of the above, resolved per section
            by its own label (see the class docstring).
        bifurcation_bias : BifurcationBias or dict, optional
            Bifurcation bias passed to each MorphologySectionSynthesizer.
            May also be a dict mapping label to bias.
        centrifugal : bool or dict, default True
            Whether MorphologySectionSynthesizer displacement is
            centrifugal. May also be a dict mapping label to bool.
        max_angle : float or dict, default pi / 2
            Maximum angle used when sampling elongation directions. May
            also be a dict mapping label to angle.
        elongation_random_weight : float or dict, default 1.0
            Weight of the random elongation component. May also be a
            dict mapping label to weight.
        elongation_bias_weight : float or dict, default 1.0
            Global weight applied to elongation biases. May also be a
            dict mapping label to weight.
<<<<<<< HEAD
        correction_type : None, "somatic", "somatodendritic", or dict, optional
=======
        correction_type : None, "somatic", "somatodendritic", "root", or dict, optional
>>>>>>> 21a7a56 (last version)
            Direction-correction mode applied by each
            MorphologySectionSynthesizer during elongation (see
            ``neuwalk.core.morphology.SectionSynthesizer``).
            ``"somatodendritic"`` requires ``axis_direction`` to be set
<<<<<<< HEAD
            for the same label. May also be a dict mapping label to one
            of these three values, e.g.::
=======
            for the same label. ``"root"`` corrects toward the section's
            own label-group root (the nearest ancestor with no parent,
            or whose parent has a different label) rather than the
            overall soma -- useful for a section grafted onto another
            tree, such as an oblique. May also be a dict mapping label
            to one of these four values, e.g.::
>>>>>>> 21a7a56 (last version)

                correction_type={
                    "basal_dendrite": "somatic",
                    "apical_dendrite": "somatodendritic",
                }

        parent : Section, optional
            Existing section that the primary sections will be connected
            to directly, as independent roots. This is unrelated to
            ``soma``: no soma is created, and the primary sections are
            exposed through ``self.roots`` instead. As an extra check, if
            ``parent.root`` turns out to already be a soma section,
            ``self.soma`` is set to it. If ``parent`` is not given
            (default), the original soma-based behavior applies: a soma
            is created (or reused via ``synthesize(soma=...)``) and the
            primary sections are attached to it.
        """
        # in this case we have multiple roots,
        # that are accepted only if merged into a soma
        if type(topology) == list:
            topology = self._merge_profiles(topology)

            
        if not isinstance(topology, TopologySectionSynthesizer):
            raise TypeError("topology must be a TopologySectionSynthesizer.")

        if not isinstance(rng, Random):
            raise TypeError("rng must be a Random instance.")

        if origin is None:
            origin = np.zeros(3, dtype=float)

        origin = np.asarray(origin, dtype=float)

        if origin.shape != (3,):
            raise ValueError("origin must have shape (3,).")

        # axis_direction may be a dict (resolved per label later); only
        # validate it now when it is an actual vector.
        if axis_direction is not None and not isinstance(axis_direction, dict):
            axis_direction = np.asarray(axis_direction, dtype=float)

            if axis_direction.shape != (3,):
                raise ValueError("axis_direction must have shape (3,).")

            if np.isclose(np.linalg.norm(axis_direction), 0.0):
                raise ValueError("axis_direction cannot be the zero vector.")

        if parent is not None and not isinstance(parent, Section):
            raise TypeError("parent must be a Section.")

        self.topology = topology
        self.rng = rng
        self.origin = origin.copy()
        self.theta = theta
        self.phi = phi
        self.axis_direction = (
            axis_direction if axis_direction is None or isinstance(axis_direction, dict)
            else axis_direction.copy()
        )
        self.elongation_bias = elongation_bias
        self.bifurcation_bias = bifurcation_bias
        self.bifurcation_internal_bias = bifurcation_internal_bias
        self.centrifugal = bool(centrifugal) if not isinstance(centrifugal, dict) else centrifugal
        self.max_angle = max_angle
        self.elongation_random_weight = elongation_random_weight
        self.elongation_bias_weight = elongation_bias_weight
        self.correction_type = correction_type
        self.active_sections = {}
        self.parent = parent
        self.soma = None
        self.roots = [] if parent is not None else None
        self.initialized = False

        if parent is not None and parent.root.label == "soma":
            self.soma = parent.root

    @property
    def is_active(self):
        """ check if there are active sections """
        return len(self.active_sections)

    def _resolve_axis_direction(self, label):
        """Resolve and validate ``axis_direction`` for ``label``."""
        axis_direction = self._resolve(self.axis_direction, label)

        if axis_direction is None:
            return None

        axis_direction = np.asarray(axis_direction, dtype=float)

        if axis_direction.shape != (3,):
            raise ValueError("axis_direction must have shape (3,).")

        if np.isclose(np.linalg.norm(axis_direction), 0.0):
            raise ValueError("axis_direction cannot be the zero vector.")

        return axis_direction

    def _resolve_primary_angles(self, label, n):
        """Resolve theta and phi independently for the given label."""

        def _internal_resolve(angles):
            if not all(type(k) == int for k in angles.keys() if k != "default"):
                return self._resolve(angles, label)
            return angles

        theta, phi = _internal_resolve(self.theta), _internal_resolve(self.phi)

        if isinstance(theta, dict):
            theta = theta.get(n, theta.get("default"))

        if isinstance(phi, dict):
            phi = phi.get(n, phi.get("default"))

        return theta, phi

    def _next_event(self, section):
        """
        Return the next event for a profile/walk pair.

        Parameters
        ----------
        section : tuple
            Pair containing ``(TopologySectionSynthesizer, MorphologySectionSynthesizer)``.

        Returns
        -------
        str or None
            One of ``"elongate"``, ``"bifurcate"``,
            ``"bifurcate_internal"``, or ``"annihilate"``.
        """
        profile, walk = section
        generated_steps = len(walk.points) - 1

        if generated_steps < profile.step_count:
            return "elongate"

        if generated_steps > profile.step_count:
            raise RuntimeError(
                f"The MorphologySectionSynthesizer ({generated_steps}) has more steps than "
                f"the corresponding TopologySectionSynthesizer ({profile.step_count})."
            )

        if profile.children:
            if len(profile.children) != 2:
                raise RuntimeError(
                    "A bifurcation must have exactly two children."
                )

            if profile.children[0].order != profile.order or profile.children[1].order != profile.order:
                return "bifurcate_internal"

            return "bifurcate"


        return "annihilate"

    @property
    def finished(self):
        """ Check whether there are not more active sections """
        return len(self.active_sections) == 0
        
    def synthesize(self, max_steps=None, soma=None, **overrides):
        """Generate every section, processing one order at a time until none remain."""
        if max_steps is not None:
            if not isinstance(max_steps, int) or isinstance(max_steps, bool):
                raise TypeError("max_steps must be an integer or None.")

            if max_steps < 0:
                raise ValueError("max_steps cannot be negative.")


        if not self.initialized:
            if self.parent is not None:
                attachment = self.parent
            else:
                if self.soma is None:
                    if soma is not None:
                        self.soma = soma
                    else:
                        self.soma = Section(points=[self.origin.copy()], label="soma")

                attachment = self.soma

            if self.topology.label == "soma":
                primary_profiles = list(self.topology.children)
            else:
                primary_profiles = [self.topology]

            # group primary sections by label so theta/phi/axis_direction
            # can be resolved (and each group's directions generated)
            # independently per label
            profiles_by_label = {}
            for profile in primary_profiles:
                profiles_by_label.setdefault(profile.label, []).append(profile)

            ordered_profiles = []
            ordered_directions = []

            for label, profiles in profiles_by_label.items():
                theta, phi = self._resolve_primary_angles(label, len(profiles))
                directions = misc.sphere_surface_points(n=len(profiles), theta=theta, phi=phi)

                axis_direction = self._resolve_axis_direction(label)

                if axis_direction is not None:
                    directions = np.asarray([
                        misc.AxialFrame.to_global(direction, axis_direction)
                        for direction in directions
                    ])

                ordered_profiles.extend(profiles)
                ordered_directions.extend(directions)

            for profile, initial_direction in zip(ordered_profiles, ordered_directions):
                walk = MorphologySectionSynthesizer(
                    rng=self.rng,
                    step_size=profile.step_size,
                    origin=self.origin,
                    initial_direction=initial_direction,
                    elongation_bias=self._resolve(self.elongation_bias, profile.label),
                    bifurcation_bias=self._resolve(self.bifurcation_bias, profile.label),
                    bifurcation_internal_bias=self._resolve(self.bifurcation_internal_bias, profile.label),
                    centrifugal=self._resolve(self.centrifugal, profile.label),
                    parent=attachment,
                    label=profile.label,
                    max_angle=self._resolve(self.max_angle, profile.label),
                    elongation_random_weight=self._resolve(self.elongation_random_weight, profile.label),
                    elongation_bias_weight=self._resolve(self.elongation_bias_weight, profile.label),
                    axis_direction=self._resolve(self.axis_direction, profile.label),
                    correction_type=self._resolve(self.correction_type, profile.label),
                )
                self.active_sections.setdefault(profile.order, []).append((profile, walk))

                if self.roots is not None:
                    self.roots.append(walk)

            self.initialized = True

        # keep processing orders, lowest first, until none remain
        while self.active_sections:
            current_order = min(self.active_sections)

##            for _, walk in self.active_sections[current_order]:
##                # set eventual values for the sections
##                for name, value in overrides.items():
##                    setattr(walk, name, value)
##                walk.active = True

            step = 0

            while self.active_sections[current_order]:
                if max_steps is not None and step >= max_steps:
                    return self.roots if self.roots is not None else self.soma

                current_sections = self.active_sections[current_order]
                self.active_sections[current_order] = []

                for section in current_sections:
                    profile, walk = section

##                    if not walk.active:
##                        continue

                    event = self._next_event(section)

                    match event:
                        case "elongate":
                            walk.elongate()
                            self.active_sections[current_order].append(section)

                        case "bifurcate" | "bifurcate_internal":
                            # select the bifurcation
                            children = walk.bifurcate() if event == "bifurcate" else walk.bifurcate_internal()

                            # handle children
                            for child_profile, child_walk in zip(profile.children, children):                                
                                    
                                # label may change
                                child_walk.label = child_profile.label

                                # re-resolve every walk-level parameter for the
                                # (possibly new) label, rather than silently
                                # keeping whatever the parent had
                                child_walk.elongation_bias = self._resolve(self.elongation_bias, child_walk.label)
                                child_walk.bifurcation_bias = self._resolve(self.bifurcation_bias, child_walk.label)
                                child_walk.bifurcation_internal_bias = self._resolve(self.bifurcation_internal_bias, child_walk.label)
                                child_walk.centrifugal = self._resolve(self.centrifugal, child_walk.label)
                                child_walk.max_angle = self._resolve(self.max_angle, child_walk.label)
                                child_walk.elongation_random_weight = self._resolve(self.elongation_random_weight, child_walk.label)
                                child_walk.elongation_bias_weight = self._resolve(self.elongation_bias_weight, child_walk.label)
                                child_walk.axis_direction = self._resolve(self.axis_direction, child_walk.label)
                                child_walk.correction_type = self._resolve(self.correction_type, child_walk.label)

                                if child_walk.correction_type == "somatodendritic" and child_walk.axis_direction is None:
                                    raise ValueError(
                                        f"axis_direction is required for label {child_walk.label!r} "
                                        "when its correction_type resolves to 'somatodendritic'."
                                    )

                                # add sections
                                self.active_sections.setdefault(child_profile.order, []).append((child_profile, child_walk))                  
                            
                        case "annihilate":
                            walk.annihilate()

                        case _:
                            raise RuntimeError(f"Unknown synthesis event: {event!r}.")

                for _, walk in current_sections:
                    if walk.pending_event:
                        walk.update_state()

                step += 1

            del self.active_sections[current_order]

        return self.roots if self.roots is not None else self.soma

    def describe(self):
        """Print synthesized and experimental topology statistics."""
        synthesized_primary_count = float(len(self.soma.children))
        primary_range = self.primary_count_range_constraint

        if primary_range is None:
            experimental_primary = "N/A"
        elif isinstance(primary_range, dict):
            experimental_primary = (
                f"{float(primary_range['min']):.1f}–"
                f"{float(primary_range['max']):.1f}"
            )
        else:
            experimental_primary = (
                f"{float(primary_range[0]):.1f}–"
                f"{float(primary_range[1]):.1f}"
            )

        print(
            "Initial primary sections: "
            f"synthesized={synthesized_primary_count:.1f}, "
            f"experimental={experimental_primary}"
        )

        synthesized_bifurcation_count = float(
            self.soma.bifurcation_count()
        )
        bifurcation_constraint = (
            self.bifurcation_count_constraint
        )

        if bifurcation_constraint is None:
            experimental_bifurcations = "N/A"
        else:
            experimental_bifurcations = (
                f"{float(bifurcation_constraint['mean']):.1f} ± "
                f"{float(bifurcation_constraint['std']):.1f}"
            )

        print(
            "Bifurcation count: "
            f"synthesized={synthesized_bifurcation_count:.1f}, "
            f"experimental={experimental_bifurcations}"
        )

        sholl_constraint = self.sholl_plot_constraint

        if sholl_constraint is None:
            synthesized_sholl = np.asarray(
                self.soma.sholl_plot(
                    bin_size=self.bin_size,
                ),
                dtype=float,
            )

            print("Sholl plot:")
            print("radius  synthesized")

            for index, synthesized in enumerate(
                synthesized_sholl
            ):
                radius = index * self.bin_size
                print(
                    f"{radius:.1f}  "
                    f"{synthesized:.1f}"
                )

            return

        experimental_mean = np.asarray(
            sholl_constraint["mean"],
            dtype=float,
        )
        experimental_std = np.asarray(
            sholl_constraint["std"],
            dtype=float,
        )

        if experimental_mean.ndim != 1 or experimental_std.shape != experimental_mean.shape:
            raise ValueError(
                "Sholl mean and standard deviation must be "
                "one-dimensional arrays with equal shape."
            )

        max_distance = (
            max(len(experimental_mean) - 1, 0)
            * self.bin_size
        )
        synthesized_sholl = np.asarray(
            self.soma.sholl_plot(
                bin_size=self.bin_size,
                max_distance=max_distance,
            ),
            dtype=float,
        )

        print("Sholl plot:")
        print(
            "radius  synthesized  "
            "experimental mean  experimental std"
        )

        for index, (
            synthesized,
            mean,
            std,
        ) in enumerate(
            zip(
                synthesized_sholl,
                experimental_mean,
                experimental_std,
            )
        ):
            radius = index * self.bin_size

            print(
                f"{radius:.1f}  "
                f"{synthesized:.1f}  "
                f"{mean:.1f}  "
                f"{std:.1f}"
            )
