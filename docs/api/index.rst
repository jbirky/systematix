API reference
=============

``import systematix as sx`` re-exports the main functions and classes from the
modules below, so ``sx.get_systematics`` is :func:`systematix.core.get_systematics`,
``sx.CBV`` is :class:`systematix.detrend.CBV`, and so on.

.. list-table:: The ``sx`` namespace
   :header-rows: 1
   :widths: 30 70

   * - Topic
     - Names
   * - Systematics of a KIC target
     - :func:`~systematix.core.get_systematics`,
       :func:`~systematix.core.download_lightcurves`,
       :func:`~systematix.core.systematics_from_lightcurve`,
       :func:`~systematix.core.compute_systematics`,
       :func:`~systematix.core.parse_kic`,
       :data:`~systematix.core.HEADER_KEYS`
   * - Container
     - :class:`~systematix.observation.Observation`
   * - Sources and injection
     - :class:`~systematix.sources.Systematics`,
       :func:`~systematix.sources.inject`,
       :func:`~systematix.sources.synthetic_cadence`,
       :class:`~systematix.sources.NoSystematics`,
       :class:`~systematix.sources.PDCRatio`,
       :class:`~systematix.sources.QuietSAP`,
       :class:`~systematix.sources.DonorPool`,
       :class:`~systematix.generated.Generated`
   * - Detrenders
     - :class:`~systematix.detrend.Detrender`,
       :class:`~systematix.detrend.DetrendResult`,
       :class:`~systematix.detrend.QuarterMedian`,
       :class:`~systematix.detrend.RollingMedian`,
       :class:`~systematix.detrend.LkFlatten`,
       :class:`~systematix.detrend.Poly`,
       :class:`~systematix.detrend.Ramps`,
       :class:`~systematix.detrend.CBV`,
       :class:`~systematix.detrend.ARC2`,
       :data:`~systematix.detrend.DETRENDERS`,
       :func:`~systematix.detrend.make_detrender`
   * - Transfer functions
     - :func:`~systematix.transfer.signal_gain`,
       :func:`~systematix.transfer.transfer_sweep`,
       :func:`~systematix.transfer.plot_transfer`
   * - Donors
     - :func:`~systematix.donors.quiet_donors`,
       :func:`~systematix.donors.load_nonperiodic`
   * - S3 weight models
     - :class:`~systematix.generated.WeightModel`,
       :func:`~systematix.generated.load_weight_model`,
       :data:`~systematix.generated.KP_MATCH`
   * - Submodules
     - :mod:`~systematix.decompose`, :mod:`~systematix.rednoise`, :mod:`~systematix.steps`

.. toctree::
   :maxdepth: 1
   :caption: Modules

   core
   observation
   sources
   detrend
   transfer
   donors
   cbv
   arc2
   steps
   decompose
   rednoise
   generated
   cli
